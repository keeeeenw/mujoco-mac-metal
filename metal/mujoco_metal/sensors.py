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

"""Explicit opt-in batched Metal evaluation for a bounded builtin sensor set."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "sensors.metal"
_SUPPORTED = {
    int(mujoco.mjtSensor.mjSENS_JOINTPOS),
    int(mujoco.mjtSensor.mjSENS_JOINTVEL),
    int(mujoco.mjtSensor.mjSENS_BALLQUAT),
    int(mujoco.mjtSensor.mjSENS_BALLANGVEL),
    int(mujoco.mjtSensor.mjSENS_FRAMEPOS),
    int(mujoco.mjtSensor.mjSENS_FRAMEQUAT),
    int(mujoco.mjtSensor.mjSENS_FRAMEXAXIS),
    int(mujoco.mjtSensor.mjSENS_FRAMEYAXIS),
    int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS),
    int(mujoco.mjtSensor.mjSENS_CLOCK),
    int(mujoco.mjtSensor.mjSENS_FRAMELINVEL),
    int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
    int(mujoco.mjtSensor.mjSENS_GYRO),
    int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
    # Milestone 016 families (TACTILE/PLUGIN/USER stay deferred to 019;
    # CAMPROJECTION admitted in R07a).
    int(mujoco.mjtSensor.mjSENS_TOUCH),
    int(mujoco.mjtSensor.mjSENS_ACCELEROMETER),
    int(mujoco.mjtSensor.mjSENS_FORCE),
    int(mujoco.mjtSensor.mjSENS_TORQUE),
    int(mujoco.mjtSensor.mjSENS_MAGNETOMETER),
    int(mujoco.mjtSensor.mjSENS_RANGEFINDER),
    int(mujoco.mjtSensor.mjSENS_TENDONPOS),
    int(mujoco.mjtSensor.mjSENS_TENDONVEL),
    int(mujoco.mjtSensor.mjSENS_ACTUATORPOS),
    int(mujoco.mjtSensor.mjSENS_ACTUATORVEL),
    int(mujoco.mjtSensor.mjSENS_ACTUATORFRC),
    int(mujoco.mjtSensor.mjSENS_JOINTACTFRC),
    int(mujoco.mjtSensor.mjSENS_TENDONACTFRC),
    int(mujoco.mjtSensor.mjSENS_JOINTLIMITPOS),
    int(mujoco.mjtSensor.mjSENS_JOINTLIMITVEL),
    int(mujoco.mjtSensor.mjSENS_JOINTLIMITFRC),
    int(mujoco.mjtSensor.mjSENS_TENDONLIMITPOS),
    int(mujoco.mjtSensor.mjSENS_TENDONLIMITVEL),
    int(mujoco.mjtSensor.mjSENS_TENDONLIMITFRC),
    int(mujoco.mjtSensor.mjSENS_FRAMELINACC),
    int(mujoco.mjtSensor.mjSENS_FRAMEANGACC),
    int(mujoco.mjtSensor.mjSENS_SUBTREECOM),
    int(mujoco.mjtSensor.mjSENS_SUBTREELINVEL),
    int(mujoco.mjtSensor.mjSENS_SUBTREEANGMOM),
    int(mujoco.mjtSensor.mjSENS_INSIDESITE),
    int(mujoco.mjtSensor.mjSENS_GEOMDIST),
    int(mujoco.mjtSensor.mjSENS_GEOMNORMAL),
    int(mujoco.mjtSensor.mjSENS_GEOMFROMTO),
    int(mujoco.mjtSensor.mjSENS_CONTACT),
    int(mujoco.mjtSensor.mjSENS_E_POTENTIAL),
    int(mujoco.mjtSensor.mjSENS_E_KINETIC),
    # R07a: camera projection is pure projective math (pinned cam_project,
    # no rendering); implemented natively with fixed/body-mounted cameras.
    int(mujoco.mjtSensor.mjSENS_CAMPROJECTION),
    # Milestone 016 & 019: tactile, plugin, and user sensors
    int(mujoco.mjtSensor.mjSENS_TACTILE),
    int(mujoco.mjtSensor.mjSENS_PLUGIN),
    int(mujoco.mjtSensor.mjSENS_USER),
}
_DEFERRED_TO_019 = set()
_POS = int(mujoco.mjtStage.mjSTAGE_POS)
_VEL = int(mujoco.mjtStage.mjSTAGE_VEL)
_ACC = int(mujoco.mjtStage.mjSTAGE_ACC)
_DISABLE_SENSOR = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
_KERNEL_METADATA = (
    "sensor_type", "sensor_datatype", "sensor_needstage", "sensor_objtype",
    "sensor_objid", "sensor_reftype", "sensor_refid", "sensor_dim",
    "sensor_adr", "sensor_cutoff", "jnt_qposadr", "jnt_dofadr",
    "body_rootid", "geom_bodyid", "site_bodyid",
)
_KERNEL_FIXED_BUFFER_COUNT = 16


@dataclass(frozen=True)
class SensorDescriptor:
  """Immutable compiled metadata needed by the supported sensor kernels."""

  nsensor: int
  nsensordata: int
  nq: int
  nv: int
  njnt: int
  nbody: int
  ngeom: int
  nsite: int
  sensor_type: np.ndarray
  sensor_datatype: np.ndarray
  sensor_needstage: np.ndarray
  sensor_objtype: np.ndarray
  sensor_objid: np.ndarray
  sensor_reftype: np.ndarray
  sensor_refid: np.ndarray
  sensor_dim: np.ndarray
  sensor_adr: np.ndarray
  sensor_cutoff: np.ndarray
  sensor_intprm: np.ndarray
  jnt_type: np.ndarray
  jnt_qposadr: np.ndarray
  jnt_dofadr: np.ndarray
  body_iquat: np.ndarray
  body_rootid: np.ndarray
  body_weldid: np.ndarray
  geom_bodyid: np.ndarray
  geom_pos: np.ndarray
  geom_quat: np.ndarray
  site_bodyid: np.ndarray
  site_pos: np.ndarray
  site_quat: np.ndarray
  site_size: np.ndarray
  site_type: np.ndarray
  body_inertia: np.ndarray
  ntendon: int
  nu: int
  ncam: int
  cam_bodyid: np.ndarray
  cam_pos: np.ndarray
  cam_quat: np.ndarray
  cam_resolution: np.ndarray
  cam_fovy: np.ndarray
  cam_intrinsic: np.ndarray
  cam_sensorsize: np.ndarray
  tendon_range: np.ndarray
  tendon_margin: np.ndarray
  tendon_limited: np.ndarray
  tendon_lengthspring: np.ndarray
  tendon_stiffness: np.ndarray
  tendon_stiffnesspoly: np.ndarray
  jnt_range: np.ndarray
  jnt_margin: np.ndarray
  jnt_limited: np.ndarray
  jnt_stiffness: np.ndarray
  jnt_stiffnesspoly: np.ndarray
  qpos_spring: np.ndarray
  body_mass: np.ndarray
  body_subtreemass: np.ndarray
  body_parentid: np.ndarray
  gravity: np.ndarray
  magnetic: np.ndarray
  sensor_intprm: np.ndarray
  disableflags: int


def _vec(model, name, dtype, shape):
  value = np.asarray(getattr(model, name), dtype=dtype)
  if value.shape != shape:
    raise ValueError(f"{name} has invalid shape {value.shape}; expected {shape}")
  if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
    raise ValueError(f"{name} must be finite")
  result = np.array(value, copy=True, order="C")
  return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(shape)


def lower_sensors(model) -> SensorDescriptor:
  """Validate and snapshot sensor metadata from an MjModel (host only)."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be an MjModel")
  if model.nsensordata < 0:
    raise ValueError("invalid sensor data size")
  if max(model.nsensor, model.nsensordata, model.nq, model.nv, model.njnt, model.nbody, model.ngeom, model.nsite) > (1 << 31) - 1:
    raise ValueError("sensor model dimensions exceed the Metal int32 limit")
  ns, nj, nb, ng, nsite = (
      model.nsensor, model.njnt, model.nbody, model.ngeom, model.nsite
  )
  nt, nu = int(model.ntendon), int(model.nu)
  ncam = int(model.ncam)
  names = {
      "sensor_type": (np.int32, (ns,)),
      "sensor_datatype": (np.int32, (ns,)),
      "sensor_needstage": (np.int32, (ns,)),
      "sensor_objtype": (np.int32, (ns,)),
      "sensor_objid": (np.int32, (ns,)),
      "sensor_reftype": (np.int32, (ns,)),
      "sensor_refid": (np.int32, (ns,)),
      "sensor_dim": (np.int32, (ns,)),
      "sensor_adr": (np.int32, (ns,)),
      "sensor_cutoff": (np.float32, (ns,)),
      "sensor_intprm": (np.int32, (ns, int(mujoco.mjNSENS))),
      "jnt_type": (np.int32, (nj,)),
      "jnt_qposadr": (np.int32, (nj,)),
      "jnt_dofadr": (np.int32, (nj,)),
      "jnt_range": (np.float32, (nj, 2)),
      "jnt_margin": (np.float32, (nj,)),
      "jnt_limited": (np.bool_, (nj,)),
      "jnt_stiffness": (np.float32, (nj,)),
      "jnt_stiffnesspoly": (np.float32, (nj, 2)),
      "qpos_spring": (np.float32, (int(model.nq),)),
      "body_iquat": (np.float32, (nb, 4)),
      "body_rootid": (np.int32, (nb,)),
      "body_weldid": (np.int32, (nb,)),
      "body_parentid": (np.int32, (nb,)),
      "body_mass": (np.float32, (nb,)),
      "body_inertia": (np.float32, (nb, 3)),
      "body_subtreemass": (np.float32, (nb,)),
      "geom_bodyid": (np.int32, (ng,)),
      "geom_pos": (np.float32, (ng, 3)),
      "geom_quat": (np.float32, (ng, 4)),
      "site_bodyid": (np.int32, (nsite,)),
      "site_pos": (np.float32, (nsite, 3)),
      "site_quat": (np.float32, (nsite, 4)),
      "site_size": (np.float32, (nsite, 3)),
      "site_type": (np.int32, (nsite,)),
      "tendon_range": (np.float32, (nt, 2)),
      "tendon_margin": (np.float32, (nt,)),
      "tendon_limited": (np.bool_, (nt,)),
      "tendon_lengthspring": (np.float32, (nt, 2)),
      "tendon_stiffness": (np.float32, (nt,)),
      "tendon_stiffnesspoly": (np.float32, (nt, 2)),
      "gravity": (np.float32, (3,)),
      "magnetic": (np.float32, (3,)),
      "cam_bodyid": (np.int32, (ncam,)),
      "cam_pos": (np.float32, (ncam, 3)),
      "cam_quat": (np.float32, (ncam, 4)),
      "cam_resolution": (np.int32, (ncam, 2)),
      "cam_fovy": (np.float32, (ncam,)),
      "cam_intrinsic": (np.float32, (ncam, 4)),
      "cam_sensorsize": (np.float32, (ncam, 2)),
  }
  arrays = {name: _vec(model, name, dtype, shape) for name, (dtype, shape) in names.items() if name not in ("gravity", "magnetic")}
  for name, src in (("gravity", model.opt.gravity), ("magnetic", model.opt.magnetic)):
    value = np.asarray(src, dtype=np.float32)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
      raise ValueError(f"{name} must be finite with shape (3,)")
    result = np.array(value, copy=True, order="C")
    arrays[name] = np.frombuffer(result.tobytes(), dtype=result.dtype).reshape((3,))
  stages = {_POS, _VEL, _ACC}
  S = mujoco.mjtSensor
  _VEL_TYPES = {int(S.mjSENS_JOINTVEL), int(S.mjSENS_BALLANGVEL),
                int(S.mjSENS_FRAMELINVEL), int(S.mjSENS_FRAMEANGVEL),
                int(S.mjSENS_GYRO), int(S.mjSENS_VELOCIMETER),
                int(S.mjSENS_TENDONVEL), int(S.mjSENS_ACTUATORVEL),
                int(S.mjSENS_JOINTLIMITVEL), int(S.mjSENS_TENDONLIMITVEL),
                int(S.mjSENS_SUBTREELINVEL), int(S.mjSENS_SUBTREEANGMOM)}
  _ACC_TYPES = {int(S.mjSENS_TOUCH), int(S.mjSENS_ACCELEROMETER),
                int(S.mjSENS_FORCE), int(S.mjSENS_TORQUE),
                int(S.mjSENS_ACTUATORFRC), int(S.mjSENS_JOINTACTFRC),
                int(S.mjSENS_TENDONACTFRC), int(S.mjSENS_JOINTLIMITFRC),
                int(S.mjSENS_TENDONLIMITFRC), int(S.mjSENS_FRAMELINACC),
                int(S.mjSENS_FRAMEANGACC), int(S.mjSENS_CONTACT),
                int(S.mjSENS_TACTILE)}
  valid_objtypes = {
      int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY),
      int(mujoco.mjtObj.mjOBJ_GEOM), int(mujoco.mjtObj.mjOBJ_SITE),
      int(mujoco.mjtObj.mjOBJ_MESH),
  }
  unknown_objtypes = valid_objtypes | {int(mujoco.mjtObj.mjOBJ_UNKNOWN)}
  sensor = arrays["sensor_type"]
  for i, typ in enumerate(sensor):
    typ = int(typ)
    if typ in _DEFERRED_TO_019:
      raise ValueError(f"sensor {i}: type {typ} is deferred to milestone 019")
    if typ not in _SUPPORTED:
      raise ValueError(f"sensor {i}: unsupported MuJoCo sensor type {typ}")
    stage = int(arrays["sensor_needstage"][i])
    if typ in (int(S.mjSENS_USER), int(S.mjSENS_PLUGIN)):
      expected_stage = stage
    else:
      expected_stage = (_VEL if typ in _VEL_TYPES else
                        _ACC if typ in _ACC_TYPES else _POS)
    if stage not in stages or stage != expected_stage:
      raise ValueError(f"sensor {i}: unsupported computation stage {stage}")
    dim = int(arrays["sensor_dim"][i])
    expected_dim = 1
    if typ in (int(mujoco.mjtSensor.mjSENS_BALLQUAT), int(mujoco.mjtSensor.mjSENS_FRAMEQUAT)):
      expected_dim = 4
    elif typ in (int(mujoco.mjtSensor.mjSENS_FRAMEPOS), int(mujoco.mjtSensor.mjSENS_BALLANGVEL), int(mujoco.mjtSensor.mjSENS_FRAMELINVEL), int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL), int(mujoco.mjtSensor.mjSENS_GYRO), int(mujoco.mjtSensor.mjSENS_VELOCIMETER)) or typ in (
        int(mujoco.mjtSensor.mjSENS_FRAMEXAXIS), int(mujoco.mjtSensor.mjSENS_FRAMEYAXIS), int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS)
    ):
      expected_dim = 3
    elif typ in (int(S.mjSENS_ACCELEROMETER), int(S.mjSENS_FORCE),
                 int(S.mjSENS_TORQUE), int(S.mjSENS_FRAMELINACC),
                 int(S.mjSENS_FRAMEANGACC), int(S.mjSENS_MAGNETOMETER),
                 int(S.mjSENS_SUBTREECOM), int(S.mjSENS_SUBTREELINVEL),
                 int(S.mjSENS_SUBTREEANGMOM), int(S.mjSENS_GEOMNORMAL)):
      expected_dim = 3
    elif typ == int(S.mjSENS_GEOMFROMTO):
      expected_dim = 6
    elif typ == int(S.mjSENS_CAMPROJECTION):
      expected_dim = 2
    elif typ == int(S.mjSENS_TACTILE):
      mesh, geom = int(arrays["sensor_objid"][i]), int(arrays["sensor_refid"][i])
      if (int(arrays["sensor_objtype"][i]) != int(mujoco.mjtObj.mjOBJ_MESH)
          or not 0 <= mesh < int(model.nmesh)
          or int(arrays["sensor_reftype"][i]) != int(mujoco.mjtObj.mjOBJ_GEOM)
          or not 0 <= geom < ng):
        raise ValueError(f"sensor {i}: tactile requires a valid mesh and geom")
      count = int(model.mesh_vertnum[mesh])
      if count <= 0 or dim % count or dim // count not in (1, 2, 3):
        raise ValueError(f"sensor {i}: invalid tactile taxel/channel dimensions")
      if int(model.mesh_normalnum[mesh]) < count:
        raise ValueError(f"sensor {i}: tactile mesh is missing taxel normals")
      expected_dim = dim
    elif typ in (int(S.mjSENS_PLUGIN), int(S.mjSENS_USER)):
      expected_dim = dim
    elif typ == int(S.mjSENS_RANGEFINDER):
      expected_dim = _raydata_size(int(arrays["sensor_intprm"][i, 0]))
    elif typ == int(S.mjSENS_CONTACT):
      dataspec = int(arrays["sensor_intprm"][i, 0])
      reduce = int(arrays["sensor_intprm"][i, 1])
      if dataspec <= 0 or dataspec >= (1 << 7) or reduce not in (0, 1, 2, 3):
        raise ValueError(f"sensor {i}: invalid contact dataspec/reduce")
      expected_dim = _contact_dim(model, dataspec, arrays, i)
    if dim != expected_dim:
      raise ValueError(f"sensor {i}: expected dimension {expected_dim}, found {dim}")
    adr = int(arrays["sensor_adr"][i])
    if adr < 0 or adr + dim > model.nsensordata:
      raise ValueError(f"sensor {i}: invalid output address")
    datatype = int(arrays["sensor_datatype"][i])
    if datatype not in (int(mujoco.mjtDataType.mjDATATYPE_REAL), int(mujoco.mjtDataType.mjDATATYPE_POSITIVE), int(mujoco.mjtDataType.mjDATATYPE_AXIS), int(mujoco.mjtDataType.mjDATATYPE_QUATERNION)):
      raise ValueError(f"sensor {i}: unsupported data type {datatype}")
    if float(arrays["sensor_cutoff"][i]) < 0:
      raise ValueError(f"sensor {i}: negative cutoff is unsupported")
    if typ in (int(mujoco.mjtSensor.mjSENS_JOINTPOS), int(mujoco.mjtSensor.mjSENS_JOINTVEL), int(mujoco.mjtSensor.mjSENS_BALLQUAT), int(mujoco.mjtSensor.mjSENS_BALLANGVEL)):
      jid = int(arrays["sensor_objid"][i])
      if not 0 <= jid < nj:
        raise ValueError(f"sensor {i}: invalid joint id")
      jtype = int(arrays["jnt_type"][jid])
      if typ in (int(mujoco.mjtSensor.mjSENS_JOINTPOS), int(mujoco.mjtSensor.mjSENS_JOINTVEL)) and jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError(f"sensor {i}: joint scalar sensor requires hinge or slide joint")
      if typ in (int(mujoco.mjtSensor.mjSENS_BALLQUAT), int(mujoco.mjtSensor.mjSENS_BALLANGVEL)) and jtype != int(mujoco.mjtJoint.mjJNT_BALL):
        raise ValueError(f"sensor {i}: ball sensor requires a ball joint")
    if typ in (int(mujoco.mjtSensor.mjSENS_FRAMEPOS), int(mujoco.mjtSensor.mjSENS_FRAMEQUAT), int(mujoco.mjtSensor.mjSENS_FRAMEXAXIS), int(mujoco.mjtSensor.mjSENS_FRAMEYAXIS), int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS)):
      objtype, objid = int(arrays["sensor_objtype"][i]), int(arrays["sensor_objid"][i])
      refid, reftype = int(arrays["sensor_refid"][i]), int(arrays["sensor_reftype"][i])
      if objtype not in valid_objtypes or not 0 <= objid < _object_count(objtype, nb, ng, nsite):
        raise ValueError(f"sensor {i}: unsupported frame object type or id")
      if refid != -1 and (reftype not in valid_objtypes or not 0 <= refid < _object_count(reftype, nb, ng, nsite)):
        raise ValueError(f"sensor {i}: unsupported reference frame type or id")
    if typ in (int(mujoco.mjtSensor.mjSENS_FRAMELINVEL), int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL)):
      objtype, objid = int(arrays["sensor_objtype"][i]), int(arrays["sensor_objid"][i])
      refid, reftype = int(arrays["sensor_refid"][i]), int(arrays["sensor_reftype"][i])
      if objtype not in valid_objtypes or not 0 <= objid < _object_count(objtype, nb, ng, nsite):
        raise ValueError(f"sensor {i}: unsupported frame velocity object type or id")
      if refid != -1 and (reftype not in valid_objtypes or not 0 <= refid < _object_count(reftype, nb, ng, nsite)):
        raise ValueError(f"sensor {i}: unsupported velocity reference frame type or id")
    if typ in (int(mujoco.mjtSensor.mjSENS_GYRO), int(mujoco.mjtSensor.mjSENS_VELOCIMETER)):
      sid = int(arrays["sensor_objid"][i])
      if int(arrays["sensor_objtype"][i]) != int(mujoco.mjtObj.mjOBJ_SITE) or not 0 <= sid < nsite:
        raise ValueError(f"sensor {i}: gyro and velocimeter require a site")
      if dim != 3:
        raise ValueError(f"sensor {i}: gyro and velocimeter require dimension 3")
    if typ == int(S.mjSENS_CAMPROJECTION):
      # R07a: site target projected through a body-mounted camera (pure
      # projective math, no rendering). Fixed cameras sit on the world
      # body; tracked cameras ride their body frame.
      sid = int(arrays["sensor_objid"][i])
      if int(arrays["sensor_objtype"][i]) != int(mujoco.mjtObj.mjOBJ_SITE) or not 0 <= sid < nsite:
        raise ValueError(f"sensor {i}: camprojection requires a site target")
      cid = int(arrays["sensor_refid"][i])
      if int(arrays["sensor_reftype"][i]) != int(mujoco.mjtObj.mjOBJ_CAMERA) or not 0 <= cid < ncam:
        raise ValueError(f"sensor {i}: camprojection requires a camera reference")
      if dim != 2:
        raise ValueError(f"sensor {i}: camprojection requires dimension 2")
    if typ in (int(S.mjSENS_TENDONPOS), int(S.mjSENS_TENDONVEL),
                 int(S.mjSENS_TENDONLIMITPOS), int(S.mjSENS_TENDONLIMITVEL),
                 int(S.mjSENS_TENDONLIMITFRC), int(S.mjSENS_TENDONACTFRC)):
      tid = int(arrays["sensor_objid"][i])
      if not 0 <= tid < nt:
        raise ValueError(f"sensor {i}: invalid tendon id")
    if typ in (int(S.mjSENS_ACTUATORPOS), int(S.mjSENS_ACTUATORVEL),
                 int(S.mjSENS_ACTUATORFRC)):
      aid = int(arrays["sensor_objid"][i])
      if not 0 <= aid < nu:
        raise ValueError(f"sensor {i}: invalid actuator id")
    if typ in (int(S.mjSENS_JOINTLIMITPOS), int(S.mjSENS_JOINTLIMITVEL),
                 int(S.mjSENS_JOINTLIMITFRC), int(S.mjSENS_JOINTACTFRC)):
      jid = int(arrays["sensor_objid"][i])
      if not 0 <= jid < nj:
        raise ValueError(f"sensor {i}: invalid joint id")
    if typ in (int(S.mjSENS_SUBTREECOM), int(S.mjSENS_SUBTREELINVEL),
                 int(S.mjSENS_SUBTREEANGMOM)):
      if int(arrays["sensor_objtype"][i]) != int(mujoco.mjtObj.mjOBJ_BODY):
        raise ValueError(f"sensor {i}: subtree sensors require a body")
      if not 0 <= int(arrays["sensor_objid"][i]) < nb:
        raise ValueError(f"sensor {i}: invalid subtree body id")
    if typ == int(S.mjSENS_INSIDESITE):
      objtype = int(arrays["sensor_objtype"][i])
      if objtype not in valid_objtypes or not 0 <= int(arrays["sensor_objid"][i]) < _object_count(objtype, nb, ng, nsite):
        raise ValueError(f"sensor {i}: unsupported insidesite object")
      if (int(arrays["sensor_reftype"][i]) != int(mujoco.mjtObj.mjOBJ_SITE)
          or not 0 <= int(arrays["sensor_refid"][i]) < nsite):
        raise ValueError(f"sensor {i}: insidesite requires a site reference")
    if typ in (int(S.mjSENS_TOUCH), int(S.mjSENS_ACCELEROMETER),
                 int(S.mjSENS_FORCE), int(S.mjSENS_TORQUE),
                 int(S.mjSENS_MAGNETOMETER)):
      if int(arrays["sensor_objtype"][i]) != int(mujoco.mjtObj.mjOBJ_SITE):
        raise ValueError(f"sensor {i}: site-attached sensor requires a site")
      if not 0 <= int(arrays["sensor_objid"][i]) < nsite:
        raise ValueError(f"sensor {i}: invalid site id")
    if typ == int(S.mjSENS_RANGEFINDER):
      if int(arrays["sensor_objtype"][i]) != int(mujoco.mjtObj.mjOBJ_SITE):
        raise ValueError(f"sensor {i}: only site rangefinders are supported")
      if not 0 <= int(arrays["sensor_objid"][i]) < nsite:
        raise ValueError(f"sensor {i}: invalid rangefinder site id")
      # R07a: no SDF-scene rejection. Without a registered SDF plugin both
      # engines skip SDF geoms in rays (pinned mj_ray reports no-hit);
      # the native kernel mirrors that skip, verified by parity tests.
      # With-plugin SDF ray hits stay 019-owned.
    if typ == int(S.mjSENS_CONTACT):
      for key in ("sensor_objtype", "sensor_reftype"):
        t = int(arrays[key][i])
        if t not in unknown_objtypes:
          raise ValueError(f"sensor {i}: unsupported contact filter type")
      for key, other in (("sensor_objid", "sensor_objtype"),
                         ("sensor_refid", "sensor_reftype")):
        t = int(arrays[other][i])
        v = int(arrays[key][i])
        if t == int(mujoco.mjtObj.mjOBJ_UNKNOWN):
          if v != -1:
            raise ValueError(f"sensor {i}: unknown contact filter must use id -1")
        elif not 0 <= v < _object_count(t, nb, ng, nsite):
          raise ValueError(f"sensor {i}: invalid contact filter id")
    if typ in (int(S.mjSENS_GEOMDIST), int(S.mjSENS_GEOMNORMAL),
                 int(S.mjSENS_GEOMFROMTO)):
      for key, other in (("sensor_objid", "sensor_objtype"),
                         ("sensor_refid", "sensor_reftype")):
        t = int(arrays[other][i])
        if t not in (int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_GEOM)):
          raise ValueError(f"sensor {i}: geom sensors need body/geom pairs")
        if not 0 <= int(arrays[key][i]) < _object_count(t, nb, ng, nsite):
          raise ValueError(f"sensor {i}: invalid geom sensor id")
      # Mirror the coupled pair matrix: unsupported narrow-phase pairs stay
      # rejected here instead of silently reporting cutoff. SDF-involved
      # pairs have plugin-defined ray/distance paths (019).
      from mujoco_metal.coupled_constraints import pair_max_contacts
      _sdf_ip = int(model.opt.sdf_initpoints)
      _SDF_T = int(mujoco.mjtGeom.mjGEOM_SDF)
      for _ga in _sensor_geoms(model, int(arrays["sensor_objtype"][i]),
                               int(arrays["sensor_objid"][i])):
        for _gb in _sensor_geoms(model, int(arrays["sensor_reftype"][i]),
                                 int(arrays["sensor_refid"][i])):
          _ta, _tb = int(model.geom_type[_ga]), int(model.geom_type[_gb])
          if _ga == _gb:
            raise ValueError(f"sensor {i}: geom pair must be different")
          if _ta == _SDF_T or _tb == _SDF_T:
            raise ValueError(
                f"sensor {i}: SDF geom distance is unsupported in 016")
          pair_max_contacts(_ta, _tb, _sdf_ip)
    if typ in (int(S.mjSENS_FRAMELINACC), int(S.mjSENS_FRAMEANGACC)):
      objtype, objid = int(arrays["sensor_objtype"][i]), int(arrays["sensor_objid"][i])
      if objtype not in valid_objtypes or objtype == int(mujoco.mjtObj.mjOBJ_UNKNOWN) or not 0 <= objid < _object_count(objtype, nb, ng, nsite):
        raise ValueError(f"sensor {i}: unsupported frame acceleration object")
    if typ in (int(S.mjSENS_E_POTENTIAL), int(S.mjSENS_E_KINETIC)):
      pass
    if int(arrays["sensor_datatype"][i]) in (int(mujoco.mjtDataType.mjDATATYPE_AXIS), int(mujoco.mjtDataType.mjDATATYPE_QUATERNION)) and float(arrays["sensor_cutoff"][i]) > 0:
      # MuJoCo cutoff is ignored for these normalized data types.
      pass
  arrays["ntendon"] = nt
  arrays["nu"] = nu
  arrays["ncam"] = ncam
  # Timing is applied by the canonical DeviceHistory stage. Pinned 3.10
  # retains sensor_noise as metadata but does not sample noise in mj_sensor.
  from mujoco_metal.history import lower_history
  lower_history(model)
  return SensorDescriptor(
      model.nsensor, model.nsensordata, model.nq, model.nv, nj, nb, ng, nsite,
      **arrays, disableflags=int(model.opt.disableflags),
  )


def _sensor_geoms(model, objtype, objid):
  """Geom ids covered by a geom-sensor object (body expands to its geoms)."""
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_GEOM):
    return [int(objid)]
  adr, num = int(model.body_geomadr[objid]), int(model.body_geomnum[objid])
  return [adr + k for k in range(num)]


def _object_count(objtype, nbody, ngeom, nsite):
  if objtype in (int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY)):
    return nbody
  if objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
    return ngeom
  return nsite


_CONDATA_SIZE = (1, 3, 3, 1, 3, 3, 3)
_RAYDATA_SIZE = (1, 3, 3, 3, 3, 1)


def _raydata_size(dataspec):
  if dataspec <= 0 or dataspec >= (1 << 6):
    raise ValueError(f"invalid rangefinder dataspec {dataspec}")
  return sum(s for b, s in enumerate(_RAYDATA_SIZE) if dataspec & (1 << b))


def _contact_dim(model, dataspec, arrays, i):
  size = sum(s for b, s in enumerate(_CONDATA_SIZE) if dataspec & (1 << b))
  if size <= 0:
    raise ValueError(f"sensor {i}: empty contact dataspec")
  # The 3.10 compiler emits exactly one slot (dim == slot size); reductions
  # select or aggregate matches into it.
  if int(arrays["sensor_dim"][i]) != size:
    raise ValueError(
        f"sensor {i}: multi-slot contact sensors are unsupported")
  return size


def _transform_spatial(vec, newpos, oldpos, flg_force):
  # Pinned mju_transformSpatial with identity rotation (global frame kept).
  vec = np.asarray(vec, dtype=np.float64)
  dif = np.asarray(newpos, dtype=np.float64) - np.asarray(oldpos, dtype=np.float64)
  out = vec.copy()
  if flg_force:
    out[:3] = vec[:3] - np.cross(dif, vec[3:])
  else:
    out[3:] = vec[3:] - np.cross(dif, vec[:3])
  return out


def _ray_quad(a, b, c):
  # Pinned ray_quad: solve a*x^2 + 2*b*x + c = 0, smallest nonneg root or -1.
  disc = b * b - a * c
  if disc < 0 or abs(a) < 1e-30:
    return -1.0
  root = np.sqrt(disc)
  x0, x1 = (-b - root) / a, (-b + root) / a
  if x0 >= 0:
    return x0
  if x1 >= 0:
    return x1
  return -1.0


def _ray_map(pos, mat, pnt, vec):
  # Pinned ray_map: rigid transform into geom local frame.
  m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
  lpnt = m.T @ (np.asarray(pnt, dtype=np.float64) - np.asarray(pos, dtype=np.float64))
  lvec = m.T @ np.asarray(vec, dtype=np.float64)
  return lpnt, lvec


def _ray_geom(pos, mat, size, pnt, vec, geom_type):
  # Pinned mju_rayGeom for analytic site-zone shapes (touch sensors).
  G = mujoco.mjtGeom
  size = np.asarray(size, dtype=np.float64)
  t = int(geom_type)
  if t == int(G.mjGEOM_SPHERE):
    dif = np.asarray(pnt) - np.asarray(pos)
    a = float(np.dot(vec, vec))
    b = float(np.dot(vec, dif))
    return _ray_quad(a, b, float(np.dot(dif, dif)) - size[0] ** 2)
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)
  if t == int(G.mjGEOM_BOX):
    ssz = float(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)
    if _ray_geom(pos, None, np.array([np.sqrt(ssz)]), pnt, vec, G.mjGEOM_SPHERE) < 0:
      return -1.0
    faces = ((1, 2), (0, 2), (0, 1))
    best = -1.0
    for k in range(3):
      if abs(lvec[k]) <= mujoco.mjMINVAL:
        continue
      for side in (-1, 1):
        sol = (side * size[k] - lpnt[k]) / lvec[k]
        if sol >= 0:
          p0 = lpnt[faces[k][0]] + sol * lvec[faces[k][0]]
          p1 = lpnt[faces[k][1]] + sol * lvec[faces[k][1]]
          if abs(p0) <= size[faces[k][0]] and abs(p1) <= size[faces[k][1]]:
            if best < 0 or sol < best:
              best = sol
    return best
  if t == int(G.mjGEOM_PLANE):
    if lvec[2] > -mujoco.mjMINVAL:
      return -1.0
    x = -lpnt[2] / lvec[2]
    if x < 0:
      return -1.0
    p0, p1 = lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1]
    if (size[0] <= 0 or abs(p0) <= size[0]) and (size[1] <= 0 or abs(p1) <= size[1]):
      return x
    return -1.0
  if t == int(G.mjGEOM_CAPSULE):
    ssz = size[0] + size[1]
    if _ray_geom(pos, None, np.array([ssz]), pnt, vec, G.mjGEOM_SPHERE) < 0:
      return -1.0
    best = -1.0
    a = lvec[0] ** 2 + lvec[1] ** 2
    b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
    c = lpnt[0] ** 2 + lpnt[1] ** 2 - size[0] ** 2
    sol = _ray_quad(a, b, c)
    if sol >= 0 and abs(lpnt[2] + sol * lvec[2]) <= size[1]:
      best = sol
    a2 = float(np.dot(lvec, lvec))
    for cap in (size[1], -size[1]):
      ld = lpnt - np.array([0.0, 0.0, cap])
      b2 = float(np.dot(lvec, ld))
      c2 = float(np.dot(ld, ld)) - size[0] ** 2
      for xx in _ray_quad_roots(a2, b2, c2):
        if xx >= 0 and (lpnt[2] + xx * lvec[2] >= size[1] if cap > 0
                        else lpnt[2] + xx * lvec[2] <= -size[1]):
          if best < 0 or xx < best:
            best = xx
    return best
  if t == int(G.mjGEOM_ELLIPSOID):
    s = 1.0 / (size * size)
    a = float(np.dot(s * lvec, lvec))
    b = float(np.dot(s * lvec, lpnt))
    c = float(np.dot(s * lpnt, lpnt)) - 1.0
    return _ray_quad(a, b, c)
  if t == int(G.mjGEOM_CYLINDER):
    ssz = size[0] ** 2 + size[1] ** 2
    if _ray_geom(pos, None, np.array([np.sqrt(ssz)]), pnt, vec, G.mjGEOM_SPHERE) < 0:
      return -1.0
    best = -1.0
    if abs(lvec[2]) > mujoco.mjMINVAL:
      for side in (-1, 1):
        sol = (side * size[1] - lpnt[2]) / lvec[2]
        if sol >= 0:
          p0 = lpnt[0] + sol * lvec[0]
          p1 = lpnt[1] + sol * lvec[1]
          if p0 * p0 + p1 * p1 <= size[0] ** 2 and (best < 0 or sol < best):
            best = sol
    a = lvec[0] ** 2 + lvec[1] ** 2
    b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
    c = lpnt[0] ** 2 + lpnt[1] ** 2 - size[0] ** 2
    sol = _ray_quad(a, b, c)
    if sol >= 0 and abs(lpnt[2] + sol * lvec[2]) <= size[1]:
      if best < 0 or sol < best:
        best = sol
    return best
  return -1.0


def _ray_quad_roots(a, b, c):
  disc = b * b - a * c
  if disc < 0 or abs(a) < 1e-30:
    return []
  root = np.sqrt(disc)
  return [(-b - root) / a, (-b + root) / a]


def _check_match(desc, body, geom, objtype, objid):
  # Pinned checkMatch (rigid bodies only; flex absent by admission).
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_UNKNOWN):
    return True
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_SITE):
    return True
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_GEOM):
    return int(objid) == int(geom)
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_BODY):
    return int(objid) == int(body)
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_XBODY):
    b = int(body)
    while b > int(objid):
      b = int(np.asarray(desc.body_parentid)[b])
    return b == int(objid)
  return False


def _match_contact(desc, poses, w, ex, j, type1, id1, type2, id2):
  # Pinned matchContact; returns 0/1/-1. Site filter uses inside-geom.
  U = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
  if type1 == U and type2 == U:
    return 1
  if type1 == int(mujoco.mjtObj.mjOBJ_SITE):
    sp = np.asarray(poses["site_pos"])[w, id1]
    sq = _unit(np.asarray(poses["site_quat"])[w, id1], "site quaternion")
    if not _inside_geom(sp, _quat_mat(sq), np.asarray(desc.site_size)[id1],
                        int(np.asarray(desc.site_type)[id1]),
                        np.asarray(ex["contact_pos"][w, j], dtype=np.float64)):
      return 0
  g1, g2 = int(ex["contact_geom"][w, j, 0]), int(ex["contact_geom"][w, j, 1])
  b1 = int(desc.geom_bodyid[g1]) if g1 >= 0 else -1
  b2 = int(desc.geom_bodyid[g2]) if g2 >= 0 else -1
  m11 = _check_match(desc, b1, g1, type1, id1)
  m12 = _check_match(desc, b2, g2, type1, id1)
  m21 = _check_match(desc, b1, g1, type2, id2)
  m22 = _check_match(desc, b2, g2, type2, id2)
  if not (m11 or m12) or not (m21 or m22):
    return 0
  if type1 != U and type2 != U:
    reg = m11 and m22
    rev = m12 and m21
    if reg and not rev:
      return 1
    if rev and not reg:
      return -1
    if reg and rev:
      return 1
  elif type1 != U:
    return 1 if m11 else -1
  elif type2 != U:
    return 1 if m22 else -1
  return 0


def _fill_contact_slot(value, dataspec, nfound, force, torque, dist, pos,
                       normal, tangent, flip=1):
  # Pinned copySensorData single-slot layout with flip handling.
  off = 0
  if dataspec & 1:
    value[off] = nfound
    off += 1
  if dataspec & 2:
    f = np.asarray(force, dtype=np.float64).copy()
    if flip < 0:
      f[2] *= -1
    value[off:off + 3] = f
    off += 3
  if dataspec & 4:
    t = np.asarray(torque, dtype=np.float64).copy()
    if flip < 0:
      t[2] *= -1
    value[off:off + 3] = t
    off += 3
  if dataspec & 8:
    value[off] = dist
    off += 1
  if dataspec & 16:
    value[off:off + 3] = normal
    off += 3
  if dataspec & 32:
    value[off:off + 3] = tangent
    off += 3


def _quat_mat(q):
  q = _unit(np.asarray(q, dtype=np.float64), "quaternion")
  w, x, y, z = q
  return np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def _inside_geom(pos, mat9, size, geom_type, point):
  # Pinned mju_insideGeom (3.10.0); mat9 is row-major 3x3.
  G = mujoco.mjtGeom
  vec = np.asarray(point, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
  if int(geom_type) == int(G.mjGEOM_SPHERE):
    return bool(np.dot(vec, vec) < size[0] * size[0])
  mat = np.asarray(mat9, dtype=np.float64).reshape(3, 3)
  plocal = mat.T @ vec
  if int(geom_type) == int(G.mjGEOM_CAPSULE):
    zc = min(max(plocal[2], -size[1]), size[1])
    return bool(plocal[0] ** 2 + plocal[1] ** 2 + (plocal[2] - zc) ** 2 < size[0] ** 2)
  if int(geom_type) == int(G.mjGEOM_ELLIPSOID):
    return bool(sum(plocal[k] ** 2 / size[k] ** 2 for k in range(3)) < 1)
  if int(geom_type) == int(G.mjGEOM_CYLINDER):
    return bool(abs(plocal[2]) < size[1]
                and plocal[0] ** 2 + plocal[1] ** 2 < size[0] ** 2)
  if int(geom_type) == int(G.mjGEOM_BOX):
    return bool(all(abs(plocal[k]) < size[k] for k in range(3)))
  if int(geom_type) == int(G.mjGEOM_PLANE):
    return bool(plocal[2] < 0)
  return False


def _quat_mul(a, b):
  return np.array([a[0]*b[0] - np.dot(a[1:], b[1:]), *(a[0]*b[1:] + b[0]*a[1:] + np.cross(a[1:], b[1:]))])


def _quat_rot(q, v):
  return v + 2 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


def _unit(q, name):
  norm = np.linalg.norm(q)
  if not np.isfinite(norm) or norm <= 0:
    raise ValueError(f"{name} must be a nonzero quaternion")
  return q / norm


def _quat_mul(a, b):
  a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
  return np.array([a[0] * b[0] - np.dot(a[1:], b[1:]),
                   *(a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]))])


def _cam_project_reference(model, site_pos, body_pos, body_quat, camid):
  """Exact host port of pinned ``cam_project`` (R07a, no rendering).

  Camera world frame composes the mount-body frame with the compiled
  local offset, matching ``d->cam_xpos/xmat``; then the pinned
  translation/rotation/focal/image chain projects the site target.
  """
  site_pos = np.asarray(site_pos, dtype=np.float64)
  b = int(np.asarray(model.cam_bodyid[camid]))
  bq = _unit(np.asarray(body_quat[b], dtype=np.float64), "body quaternion")
  bp = np.asarray(body_pos[b], dtype=np.float64)
  lq = _unit(np.asarray(model.cam_quat[camid], dtype=np.float64), "camera quaternion")
  wq = _quat_mul(bq, lq)
  campos = bp + _quat_rot(bq, np.asarray(model.cam_pos[camid], dtype=np.float64))
  res = np.asarray(model.cam_resolution[camid], dtype=np.float64)
  fovy = float(np.asarray(model.cam_fovy[camid]))
  intrinsic = np.asarray(model.cam_intrinsic[camid], dtype=np.float64)
  sensorsize = np.asarray(model.cam_sensorsize[camid], dtype=np.float64)
  # Rotation block: rotation[i][j] = xmat[j*3+i] with column-major xmat.
  x, y, z, w = wq[1], wq[2], wq[3], wq[0]
  xmat = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
  if sensorsize[0] and sensorsize[1]:
    fx = intrinsic[0] / sensorsize[0] * res[0]
    fy = intrinsic[1] / sensorsize[1] * res[1]
  else:
    fx = fy = 0.5 / np.tan(fovy * np.pi / 360.0) * res[1]
  # Direct chain evaluation: translation shifts by -campos, rotation
  # (rotation[i][j] = xmat[j*3+i], i.e. the xmat matrix itself, verified
  # against CPU sensordata) contracts it, focal scales, image centers.
  p = site_pos - campos
  q = xmat @ p
  px = np.array([-fx * q[0], fy * q[1], q[2]])
  img = np.array([px[0] + res[0] / 2 * px[2], px[1] + res[1] / 2 * px[2], px[2]])
  denom = img[2]
  if abs(denom) < float(mujoco.mjMINVAL):
    denom = min(denom, -float(mujoco.mjMINVAL)) if denom < 0 else max(denom, float(mujoco.mjMINVAL))
  return np.array([img[0] / denom, img[1] / denom])


def _poly_potential(linear, poly, x):
  return 0.5 * linear * x * x + poly[0] / 3 * x**3 + poly[1] / 4 * x**4


def sensor_oracle(model, qpos, qvel, time, poses, sensordata=None, stages=(_POS, _VEL), extra=None):
  """Float64 CPU reference for the supported stage; accepts FK pose arrays.

  `extra` optionally carries host-side pinned quantities for families whose
  inputs are not in `poses`: tendon/actuator lengths and velocities,
  subtree aggregates, energies, and efc rows::

      {"ten_length": (B, nt), "ten_velocity": (B, nt),
       "actuator_length": (B, nu), "actuator_velocity": (B, nu),
       "subtree_com": (B, nbody, 3), "subtree_linvel": (B, nbody, 3),
       "subtree_angmom": (B, nbody, 3), "energy": (B, 2),
       "efc_type": (B, nefc), "efc_id": (B, nefc),
       "efc_pos": (B, nefc), "efc_vel": (B, nefc),
       "efc_margin": (B, nefc), "ne": int, "nf": int}
  """
  desc = lower_sensors(model)
  qpos, qvel, time = np.asarray(qpos), np.asarray(qvel), np.asarray(time)
  if qpos.ndim != 2 or qpos.shape[0] <= 0 or qpos.shape[1] != desc.nq or qvel.shape != (len(qpos), desc.nv) or time.shape != (len(qpos),):
    raise ValueError("qpos, qvel, and time batch shapes do not match model")
  batch = len(qpos)
  out = np.zeros((batch, desc.nsensordata)) if sensordata is None else np.array(sensordata, dtype=np.float64, copy=True)
  if out.shape != (batch, desc.nsensordata):
    raise ValueError("sensordata shape does not match batch and model")
  if desc.disableflags & _DISABLE_SENSOR:
    return out
  S = mujoco.mjtSensor
  C = mujoco.mjtConstraint
  _LIMIT_JOINT = int(C.mjCNSTR_LIMIT_JOINT)
  _LIMIT_TENDON = int(C.mjCNSTR_LIMIT_TENDON)
  Need = lambda *keys: (
      {k: np.asarray(extra[k]) for k in keys} if extra is not None and all(k in extra for k in keys)
      else (_ for _ in ()).throw(ValueError(f"sensor oracle needs extra[{keys}]")))
  for w in range(batch):
    for i, raw_type in enumerate(desc.sensor_type):
      typ, stage = int(raw_type), int(desc.sensor_needstage[i])
      if stage not in stages:
        continue
      objid = int(desc.sensor_objid[i]); qa = int(desc.jnt_qposadr[objid]) if typ in (9, 18) else 0
      da = int(desc.jnt_dofadr[objid]) if typ in (10, 19) else 0
      if typ == int(mujoco.mjtSensor.mjSENS_JOINTPOS):
        value = np.array([qpos[w, qa]])
      elif typ == int(mujoco.mjtSensor.mjSENS_JOINTVEL):
        value = np.array([qvel[w, da]])
      elif typ == int(mujoco.mjtSensor.mjSENS_BALLQUAT):
        value = _unit(qpos[w, qa:qa+4], "ball quaternion")
      elif typ == int(mujoco.mjtSensor.mjSENS_BALLANGVEL):
        value = qvel[w, da:da+3]
      elif typ == int(S.mjSENS_TENDONPOS):
        value = np.array([Need("ten_length")["ten_length"][w, objid]])
      elif typ == int(S.mjSENS_TENDONVEL):
        value = np.array([Need("ten_velocity")["ten_velocity"][w, objid]])
      elif typ == int(S.mjSENS_ACTUATORPOS):
        value = np.array([Need("actuator_length")["actuator_length"][w, objid]])
      elif typ == int(S.mjSENS_ACTUATORVEL):
        value = np.array([Need("actuator_velocity")["actuator_velocity"][w, objid]])
      elif typ in (int(S.mjSENS_JOINTLIMITPOS), int(S.mjSENS_JOINTLIMITVEL)):
        ex = Need("efc_type", "efc_id", "efc_pos", "efc_vel", "efc_margin", "ne", "nf")
        value = np.array([0.0])
        for j in range(int(ex["ne"]) + int(ex["nf"]), ex["efc_type"].shape[1]):
          if (int(ex["efc_type"][w, j]) == _LIMIT_JOINT and int(ex["efc_id"][w, j]) == objid):
            raw = (ex["efc_pos"] if typ == int(S.mjSENS_JOINTLIMITPOS) else ex["efc_vel"])[w, j]
            value = np.array([raw - ex["efc_margin"][w, j] if typ == int(S.mjSENS_JOINTLIMITPOS) else raw])
            break
      elif typ in (int(S.mjSENS_TENDONLIMITPOS), int(S.mjSENS_TENDONLIMITVEL)):
        ex = Need("efc_type", "efc_id", "efc_pos", "efc_vel", "efc_margin", "ne", "nf")
        value = np.array([0.0])
        for j in range(int(ex["ne"]) + int(ex["nf"]), ex["efc_type"].shape[1]):
          if (int(ex["efc_type"][w, j]) == _LIMIT_TENDON and int(ex["efc_id"][w, j]) == objid):
            raw = (ex["efc_pos"] if typ == int(S.mjSENS_TENDONLIMITPOS) else ex["efc_vel"])[w, j]
            value = np.array([raw - ex["efc_margin"][w, j] if typ == int(S.mjSENS_TENDONLIMITPOS) else raw])
            break
      elif typ == int(S.mjSENS_MAGNETOMETER):
        _site_q = _unit(np.asarray(poses["site_quat"])[w, objid], "site quaternion")
        value = _quat_rot(np.array([_site_q[0], *(-_site_q[1:])]),
                          np.asarray(desc.magnetic, dtype=np.float64))
      elif typ == int(S.mjSENS_CAMPROJECTION):
        value = _cam_project_reference(
            model, np.asarray(poses["site_pos"])[w, objid],
            np.asarray(poses["body_pos"])[w], np.asarray(poses["body_quat"])[w],
            int(desc.sensor_refid[i]))
      elif typ == int(S.mjSENS_SUBTREECOM):
        value = np.asarray(Need("subtree_com")["subtree_com"][w, objid]).copy()
      elif typ == int(S.mjSENS_SUBTREELINVEL):
        value = np.asarray(Need("subtree_linvel")["subtree_linvel"][w, objid]).copy()
      elif typ == int(S.mjSENS_SUBTREEANGMOM):
        value = np.asarray(Need("subtree_angmom")["subtree_angmom"][w, objid]).copy()
      elif typ == int(S.mjSENS_INSIDESITE):
        otype = int(desc.sensor_objtype[i])
        if otype == int(mujoco.mjtObj.mjOBJ_BODY):
          p = np.asarray(poses["inertial_pos"])[w, objid]
          if (objid > 0 and float(np.asarray(desc.body_mass)[objid]) < mujoco.mjMINVAL
              and float(np.asarray(desc.body_subtreemass)[objid]) >= mujoco.mjMINVAL):
            p = np.asarray(Need("subtree_com")["subtree_com"][w, objid])
        else:
          p, _ = _frame_pose(desc, poses, w, otype, objid)
        refid = int(desc.sensor_refid[i])
        sp = np.asarray(poses["site_pos"])[w, refid]
        sq = _unit(np.asarray(poses["site_quat"])[w, refid], "site quaternion")
        value = np.array([float(_inside_geom(
            sp, _quat_mat(sq), np.asarray(desc.site_size)[refid],
            int(np.asarray(desc.site_type)[refid]), np.asarray(p, dtype=np.float64)))])
      elif typ in (int(S.mjSENS_E_POTENTIAL), int(S.mjSENS_E_KINETIC)):
        value = np.array([float(Need("energy")["energy"][w, 0 if typ == int(S.mjSENS_E_POTENTIAL) else 1])])
      elif typ == int(S.mjSENS_RANGEFINDER):
        ex = Need("ray")
        r = np.asarray(ex["ray"][w, i], dtype=np.float64)
        dataspec = int(desc.sensor_intprm[i, 0])
        sid = objid
        origin = np.asarray(poses["site_pos"])[w, sid].copy()
        sq = _unit(np.asarray(poses["site_quat"])[w, sid], "site quaternion")
        direction = _quat_rot(sq, np.array([0.0, 0.0, 1.0]))
        hit = float(r[0]) >= 0
        parts = []
        if dataspec & 1:
          parts += [float(r[0])]
        if dataspec & 2:
          parts += list(direction if hit else np.zeros(3))
        if dataspec & 4:
          parts += list(origin)
        point = origin + direction * float(r[0]) if hit else np.zeros(3)
        if dataspec & 8:
          parts += list(point)
        if dataspec & 16:
          parts += list(r[4:7] if hit else np.zeros(3))
        if dataspec & 32:
          parts += [float(r[0]) if hit else -1.0]
        value = np.array(parts)
      elif typ in (int(S.mjSENS_GEOMDIST), int(S.mjSENS_GEOMNORMAL),
                    int(S.mjSENS_GEOMFROMTO)):
        ex = Need("geom")
        g = np.asarray(ex["geom"][w, i], dtype=np.float64)
        if typ == int(S.mjSENS_GEOMDIST):
          value = np.array([float(g[0])])
        elif typ == int(S.mjSENS_GEOMNORMAL):
          seg = g[4:7] - g[1:4]
          n = float(np.linalg.norm(seg))
          value = seg / n if n > 0 else np.zeros(3)
        else:
          value = g[1:7].copy()
      elif typ == int(S.mjSENS_CONTACT):
        ex = Need("contact_geom", "contact_frame", "contact_pos",
                  "contact_dist", "contact_force", "contact_efc", "ncon")
        dataspec = int(desc.sensor_intprm[i, 0])
        reduce = int(desc.sensor_intprm[i, 1])
        reftype, refid = int(desc.sensor_reftype[i]), int(desc.sensor_refid[i])
        otype, oid = int(desc.sensor_objtype[i]), objid
        BEST = None
        order = []
        for j in range(int(ex["ncon"][w])):
          if int(ex["contact_efc"][w, j]) < 0:
            continue
          m = _match_contact(desc, poses, w, ex, j, otype, oid, reftype, refid)
          if not m:
            continue
          order.append((j, m))
        crit = []
        for (j, m) in order:
          if reduce == 1:
            c = float(ex["contact_dist"][w, j])
          elif reduce == 2:
            f = np.asarray(ex["contact_force"][w, j], dtype=np.float64)
            c = -float(np.dot(f[:3], f[:3]))
          else:
            c = 0.0
          crit.append(c)
        value = np.zeros(int(desc.sensor_dim[i]))
        if reduce == 3:
          # Net-force aggregation over all matches.
          F = np.zeros(3)
          T = np.zeros(3)
          P = np.zeros(3)
          tot = 0.0
          for (j, m) in order:
            f = np.asarray(ex["contact_force"][w, j], dtype=np.float64)[:3]
            t = np.asarray(ex["contact_force"][w, j], dtype=np.float64)[3:6]
            fr = np.asarray(ex["contact_frame"][w, j]).reshape(3, 3)
            F += fr.T @ f
            T += fr.T @ t
            p = np.asarray(ex["contact_pos"][w, j], dtype=np.float64)
            wgt = float(np.linalg.norm(np.concatenate((f, t))))
            P += wgt * p
            tot += wgt
          if tot > 0:
            P /= tot
          for (j, m) in order:
            f = np.asarray(ex["contact_force"][w, j], dtype=np.float64)[:3]
            t = np.asarray(ex["contact_force"][w, j], dtype=np.float64)[3:6]
            fr = np.asarray(ex["contact_frame"][w, j]).reshape(3, 3)
            T += np.cross(np.asarray(ex["contact_pos"][w, j], dtype=np.float64) - P,
                          fr.T @ f)
          _fill_contact_slot(value, dataspec, len(order), F, T, 0.0, P,
                             np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        else:
          pick = None
          if order and reduce == 0:
            pick = order[0]
          elif order:
            k = int(np.argmin(np.asarray(crit)))
            pick = order[k]
          nmatch = len(order)
          if pick is not None:
            j, m = pick
            f = np.asarray(ex["contact_force"][w, j], dtype=np.float64)
            _fill_contact_slot(
                value, dataspec, nmatch, f[:3].copy(), f[3:6].copy(),
                float(ex["contact_dist"][w, j]),
                np.asarray(ex["contact_pos"][w, j], dtype=np.float64),
                np.asarray(ex["contact_frame"][w, j]).reshape(3, 3)[0] * m,
                np.asarray(ex["contact_frame"][w, j]).reshape(3, 3)[1] * m,
                flip=m)
          else:
            _fill_contact_slot(value, dataspec, 0, np.zeros(3), np.zeros(3),
                               0.0, np.zeros(3), np.zeros(3), np.zeros(3))
      elif typ == int(S.mjSENS_TOUCH):
        ex = Need("contact_geom", "contact_frame", "contact_pos",
                  "contact_dist", "contact_force", "contact_efc", "ncon")
        sid = objid
        sbody = int(desc.site_bodyid[sid])
        total = 0.0
        for j in range(int(ex["ncon"][w])):
          if int(ex["contact_efc"][w, j]) < 0:
            continue
          g0, g1 = int(ex["contact_geom"][w, j, 0]), int(ex["contact_geom"][w, j, 1])
          b0 = int(desc.geom_bodyid[g0]) if g0 >= 0 else -1
          b1 = int(desc.geom_bodyid[g1]) if g1 >= 0 else -1
          if sbody != b0 and sbody != b1:
            continue
          fn = float(ex["contact_force"][w, j, 0])
          if fn <= 0:
            continue
          frame = np.asarray(ex["contact_frame"][w, j]).reshape(3, 3)
          ray = frame[0] * fn
          nray = ray / max(np.linalg.norm(ray), 1e-30)
          if sbody == b1:
            nray = -nray
          sp = np.asarray(poses["site_pos"])[w, sid]
          sq = _unit(np.asarray(poses["site_quat"])[w, sid], "site quaternion")
          sm = _quat_mat(sq)
          if _ray_geom(sp, sm, np.asarray(desc.site_size)[sid],
                       np.asarray(ex["contact_pos"][w, j]), nray,
                       int(np.asarray(desc.site_type)[sid])) >= 0:
            total += fn
        value = np.array([total])
      elif typ == int(S.mjSENS_ACCELEROMETER):
        ex = Need("cacc", "subtree_com")
        sbody = int(desc.site_bodyid[objid])
        root = int(desc.body_rootid[sbody])
        sp = np.asarray(poses["site_pos"])[w, objid]
        sub = np.asarray(ex["subtree_com"][w, root], dtype=np.float64)
        acc = np.asarray(ex["cacc"][w, sbody], dtype=np.float64)
        res = _transform_spatial(acc, sp, sub, False)
        cvel = np.asarray(poses["cvel"])[w, sbody]
        vel = _transform_spatial(cvel, sp, sub, False)
        res[3:] += np.cross(vel[:3], vel[3:])
        sq = _unit(np.asarray(poses["site_quat"])[w, objid], "site quaternion")
        value = _quat_rot(np.array([sq[0], *(-sq[1:])]), res[3:])
      elif typ in (int(S.mjSENS_FORCE), int(S.mjSENS_TORQUE)):
        ex = Need("cfrc_int", "subtree_com")
        sbody = int(desc.site_bodyid[objid])
        root = int(desc.body_rootid[sbody])
        sp = np.asarray(poses["site_pos"])[w, objid]
        sub = np.asarray(ex["subtree_com"])[w, root].astype(np.float64)
        ci = np.asarray(ex["cfrc_int"])[w, sbody].astype(np.float64)
        res = _transform_spatial(ci, sp, sub, True)
        sq = _unit(np.asarray(poses["site_quat"])[w, objid], "site quaternion")
        res[:3] = _quat_rot(np.array([sq[0], *(-sq[1:])]), res[:3])
        res[3:] = _quat_rot(np.array([sq[0], *(-sq[1:])]), res[3:])
        value = res[3:].copy() if typ == int(S.mjSENS_FORCE) else res[:3].copy()
      elif typ == int(S.mjSENS_ACTUATORFRC):
        value = np.array([float(Need("actuator_force")["actuator_force"][w, objid])])
      elif typ == int(S.mjSENS_JOINTACTFRC):
        da = int(desc.jnt_dofadr[objid])
        value = np.array([float(Need("qfrc_actuator")["qfrc_actuator"][w, da])])
      elif typ == int(S.mjSENS_TENDONACTFRC):
        ex = Need("actuator_force", "actuator_trntype", "actuator_trnid")
        frc = 0.0
        for j in range(ex["actuator_force"].shape[1]):
          if (int(ex["actuator_trntype"][j]) == int(mujoco.mjtTrn.mjTRN_TENDON)
              and int(ex["actuator_trnid"][j][0]) == objid):
            frc += float(ex["actuator_force"][w, j])
        value = np.array([frc])
      elif typ in (int(S.mjSENS_JOINTLIMITFRC), int(S.mjSENS_TENDONLIMITFRC)):
        ex = Need("efc_type", "efc_id", "efc_force", "ne", "nf")
        want_type = (_LIMIT_JOINT if typ == int(S.mjSENS_JOINTLIMITFRC)
                     else _LIMIT_TENDON)
        value = np.array([0.0])
        for j in range(int(ex["ne"]) + int(ex["nf"]), ex["efc_type"].shape[1]):
          if (int(ex["efc_type"][w, j]) == want_type and int(ex["efc_id"][w, j]) == objid):
            value = np.array([float(ex["efc_force"][w, j])])
            break
      elif typ in (int(S.mjSENS_FRAMELINACC), int(S.mjSENS_FRAMEANGACC)):
        ex = Need("cacc", "subtree_com")
        otype = int(desc.sensor_objtype[i])
        obody = objid
        if otype == int(mujoco.mjtObj.mjOBJ_GEOM):
          obody = int(desc.geom_bodyid[objid])
        elif otype == int(mujoco.mjtObj.mjOBJ_SITE):
          obody = int(desc.site_bodyid[objid])
        if int(np.asarray(desc.body_weldid)[obody]) == 0:
          value = np.zeros(3)
        else:
          root = int(desc.body_rootid[obody])
          pos, _ = _frame_pose(desc, poses, w, otype, objid)
          sub = np.asarray(ex["subtree_com"][w, root], dtype=np.float64)
          acc = np.asarray(ex["cacc"][w, obody], dtype=np.float64)
          res = _transform_spatial(acc, pos, sub, False)
          cvel = np.asarray(poses["cvel"])[w, obody]
          vel = _transform_spatial(cvel, pos, sub, False)
          res[3:] += np.cross(vel[:3], vel[3:])
          value = res[3:6].copy() if typ == int(S.mjSENS_FRAMELINACC) else res[0:3].copy()
      elif typ in (int(mujoco.mjtSensor.mjSENS_FRAMELINVEL), int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL), int(mujoco.mjtSensor.mjSENS_GYRO), int(mujoco.mjtSensor.mjSENS_VELOCIMETER)):
        if "cvel" not in poses or "root_com" not in poses:
          raise ValueError("velocity sensors require cvel and root_com in poses")
        objtype = int(desc.sensor_objtype[i])
        objid = int(desc.sensor_objid[i])
        obj_pos, obj_quat = _frame_pose(desc, poses, w, objtype, objid)
        obj_vel = _object_velocity(desc, poses, w, objtype, objid, obj_pos)
        if typ in (int(mujoco.mjtSensor.mjSENS_GYRO), int(mujoco.mjtSensor.mjSENS_VELOCIMETER)):
          rotated = np.concatenate((_quat_rot(np.array([obj_quat[0], *(-obj_quat[1:])]), obj_vel[:3]), _quat_rot(np.array([obj_quat[0], *(-obj_quat[1:])]), obj_vel[3:])))
          value = rotated[:3] if typ == int(mujoco.mjtSensor.mjSENS_GYRO) else rotated[3:]
        else:
          refid = int(desc.sensor_refid[i])
          if refid > -1:
            ref_type = int(desc.sensor_reftype[i])
            ref_pos, ref_quat = _frame_pose(desc, poses, w, ref_type, refid)
            ref_vel = _object_velocity(desc, poses, w, ref_type, refid, ref_pos)
            relative = np.concatenate((obj_vel[:3] - ref_vel[:3], obj_vel[3:] - ref_vel[3:] + np.cross(obj_pos-ref_pos, ref_vel[:3])))
            relative[:3] = _quat_rot(np.array([ref_quat[0], *(-ref_quat[1:])]), relative[:3])
            relative[3:] = _quat_rot(np.array([ref_quat[0], *(-ref_quat[1:])]), relative[3:])
            obj_vel = relative
          value = obj_vel[3:] if typ == int(mujoco.mjtSensor.mjSENS_FRAMELINVEL) else obj_vel[:3]
      elif typ == int(mujoco.mjtSensor.mjSENS_CLOCK):
        value = np.array([time[w]])
      else:
        pos, quat = _frame_pose(desc, poses, w, int(desc.sensor_objtype[i]), objid)
        refid = int(desc.sensor_refid[i])
        if refid >= 0:
          rpos, rquat = _frame_pose(desc, poses, w, int(desc.sensor_reftype[i]), refid)
        else:
          rpos, rquat = np.zeros(3), np.array([1., 0., 0., 0.])
        if typ == int(mujoco.mjtSensor.mjSENS_FRAMEPOS):
          value = _quat_rot(np.array([rquat[0], *(-rquat[1:])]), pos-rpos)
        elif typ == int(mujoco.mjtSensor.mjSENS_FRAMEQUAT):
          value = _unit(_quat_mul(np.array([rquat[0], *(-rquat[1:])]), quat), "frame quaternion")
        else:
          axis = typ - int(mujoco.mjtSensor.mjSENS_FRAMEXAXIS)
          value = _quat_rot(np.array([rquat[0], *(-rquat[1:])]), _quat_rot(quat, np.eye(3)[axis]))
      if desc.sensor_datatype[i] == int(mujoco.mjtDataType.mjDATATYPE_REAL) and desc.sensor_cutoff[i] > 0:
        value = np.clip(value, -desc.sensor_cutoff[i], desc.sensor_cutoff[i])
      elif desc.sensor_datatype[i] == int(mujoco.mjtDataType.mjDATATYPE_POSITIVE) and desc.sensor_cutoff[i] > 0:
        value = np.minimum(value, desc.sensor_cutoff[i])
      adr, dim = int(desc.sensor_adr[i]), int(desc.sensor_dim[i])
      out[w, adr:adr+dim] = value
  return out


def _frame_pose(desc, poses, world, objtype, objid):
  if objtype == int(mujoco.mjtObj.mjOBJ_BODY):
    return poses["inertial_pos"][world, objid], _unit(poses["inertial_quat"][world, objid], "inertial quaternion")
  if objtype == int(mujoco.mjtObj.mjOBJ_XBODY):
    return poses["body_pos"][world, objid], _unit(poses["body_quat"][world, objid], "body quaternion")
  if objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
    bid = int(desc.geom_bodyid[objid]); q = _quat_mul(poses["body_quat"][world, bid], desc.geom_quat[objid])
    return poses["body_pos"][world, bid] + _quat_rot(poses["body_quat"][world, bid], desc.geom_pos[objid]), _unit(q, "geom quaternion")
  bid = int(desc.site_bodyid[objid]); q = _quat_mul(poses["body_quat"][world, bid], desc.site_quat[objid])
  return poses["body_pos"][world, bid] + _quat_rot(poses["body_quat"][world, bid], desc.site_pos[objid]), _unit(q, "site quaternion")


def _object_body(desc, objtype, objid):
  if objtype in (int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY)):
    return objid
  if objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
    return int(desc.geom_bodyid[objid])
  return int(desc.site_bodyid[objid])


def _object_velocity(desc, poses, world, objtype, objid, pos):
  body = _object_body(desc, objtype, objid)
  root = int(desc.body_rootid[body])
  cvel = np.asarray(poses["cvel"]).reshape(-1, desc.nbody, 6)[world, body]
  root_com = np.asarray(poses["root_com"]).reshape(-1, desc.nbody, 3)[world, root]
  return np.concatenate((cvel[:3], cvel[3:] + np.cross(cvel[:3], pos-root_com)))


def sensor_workspace_sizes(descriptor, batch_size):
  """Check compiled workspace products before importing Torch or allocating."""
  sizes = {
      "subtree_runtime": 4 + batch_size * descriptor.nbody * 32,
      "com_scratch": batch_size * descriptor.nbody * 12,
      "sensor_output": batch_size * descriptor.nsensordata,
      "body_spatial": batch_size * descriptor.nbody * 6,
  }
  if any(size > (1 << 31)-1 for size in sizes.values()):
    raise ValueError("sensor workspace exceeds the Metal int32 index capacity")
  return sizes


class SensorProgram:
  """Batched MSL sensor evaluator. Constructor is the explicit MPS boundary."""

  def __init__(self, model, batch_size=1):
    self.descriptor = lower_sensors(model)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    if batch_size > (1 << 31) - 1 or batch_size * max(self.descriptor.nsensor, 1) > (1 << 31) - 1:
      raise ValueError("sensor dispatch exceeds the Metal int32 thread limit")
    self._workspace_sizes = sensor_workspace_sizes(self.descriptor, batch_size)
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("PyTorch MPS with compile_shader is required")
    self._torch, self.batch_size = torch, batch_size
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.evaluate_sensors
    d = self.descriptor
    self._meta = {}
    for name in ("sensor_type", "sensor_datatype", "sensor_needstage", "sensor_objtype", "sensor_objid", "sensor_reftype", "sensor_refid", "sensor_dim", "sensor_adr", "sensor_cutoff", "jnt_type", "jnt_qposadr", "jnt_dofadr", "body_iquat", "body_rootid", "geom_bodyid", "geom_pos", "geom_quat", "site_bodyid", "site_pos", "site_quat"):
      arr = getattr(d, name)
      self._meta[name] = torch.as_tensor(np.array(arr, copy=True), device=self._device)
    self._dims = torch.tensor([d.nsensor, d.nsensordata, d.nq, d.nv, d.njnt, d.nbody, d.ngeom, d.nsite, batch_size, d.disableflags], dtype=torch.int32, device=self._device)
    self._stage_mask = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._output = torch.zeros((batch_size, d.nsensordata), dtype=torch.float32, device=self._device)
    self._needs_velocity = bool(np.any(np.isin(d.sensor_type, [
        int(mujoco.mjtSensor.mjSENS_FRAMELINVEL),
        int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
        int(mujoco.mjtSensor.mjSENS_GYRO),
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ])))
    self._mjmodel = model
    self._tactile_sensors = [i for i in range(d.nsensor)
                            if int(d.sensor_type[i]) == int(mujoco.mjtSensor.mjSENS_TACTILE)]
    self._build_state_constants(model)

  def _build_state_constants(self, model):
    """Host constants for the milestone-016 state-family kernel."""
    import struct
    torch, d = self._torch, self.descriptor
    nq, nv, nj, nb, ns = d.nq, d.nv, d.njnt, d.nbody, d.nsensor
    nt, nu, nsite = d.ntendon, d.nu, d.nsite
    J = mujoco.mjtJoint
    hinge, slide = int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE)
    meta = np.zeros((ns, 10), dtype=np.int32)
    meta[:, 0] = np.asarray(d.sensor_type, dtype=np.int32)
    meta[:, 1] = np.asarray(d.sensor_datatype, dtype=np.int32)
    meta[:, 2] = np.asarray(d.sensor_needstage, dtype=np.int32)
    meta[:, 3] = np.asarray(d.sensor_objtype, dtype=np.int32)
    meta[:, 4] = np.asarray(d.sensor_objid, dtype=np.int32)
    meta[:, 5] = np.asarray(d.sensor_dim, dtype=np.int32)
    meta[:, 6] = np.asarray(d.sensor_adr, dtype=np.int32)
    for k, c in enumerate(np.asarray(d.sensor_cutoff, dtype=np.float32)):
      meta[k, 7] = struct.unpack("<i", struct.pack("<f", float(c)))[0]
    meta[:, 8] = np.asarray(d.sensor_reftype, dtype=np.int32)
    meta[:, 9] = np.asarray(d.sensor_refid, dtype=np.int32)
    self._s_meta = torch.as_tensor(meta.copy(), dtype=torch.int32, device=self._device)
    jmeta = np.zeros((max(nj, 1), 4), dtype=np.int32)
    jlim = np.zeros((max(nj, 1), 3), dtype=np.float32)
    for j in range(nj):
      jmeta[j] = [int(model.jnt_type[j]), int(model.jnt_qposadr[j]),
                  int(model.jnt_dofadr[j]), int(bool(model.jnt_limited[j]))]
      jlim[j] = [float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]),
                 float(model.jnt_margin[j])]
    self._s_jnt_meta = torch.as_tensor(jmeta.copy(), dtype=torch.int32, device=self._device)
    self._s_jnt_lim = torch.as_tensor(jlim.copy(), dtype=torch.float32, device=self._device)
    tlim = np.zeros((max(nt, 1), 9), dtype=np.float32)
    for t in range(nt):
      tlim[t] = [float(model.tendon_range[t, 0]), float(model.tendon_range[t, 1]),
                 float(model.tendon_margin[t]),
                 float(model.tendon_lengthspring[t, 0]), float(model.tendon_lengthspring[t, 1]),
                 float(model.tendon_stiffness[t]),
                 float(model.tendon_stiffnesspoly[t, 0]), float(model.tendon_stiffnesspoly[t, 1]),
                 1.0 if bool(model.tendon_limited[t]) else 0.0]
    self._s_ten_lim = torch.as_tensor(tlim.copy(), dtype=torch.float32, device=self._device)
    massub = np.zeros((max(nb, 1), 5), dtype=np.float32)
    for b in range(nb):
      massub[b] = [float(model.body_mass[b]), float(model.body_subtreemass[b]),
                   float(model.body_inertia[b, 0]), float(model.body_inertia[b, 1]),
                   float(model.body_inertia[b, 2])]
    self._s_massub = torch.as_tensor(massub.copy(), dtype=torch.float32, device=self._device)
    tree = np.zeros((max(nb, 1), 2), dtype=np.int32)
    for b in range(nb):
      tree[b] = [int(model.body_parentid[b]), int(model.body_rootid[b])]
    self._s_body_tree = torch.as_tensor(tree.copy(), dtype=torch.int32, device=self._device)
    sg = np.zeros((max(nsite, 1), 4), dtype=np.float32)
    for s in range(nsite):
      sg[s] = [float(model.site_size[s, 0]), float(model.site_size[s, 1]),
               float(model.site_size[s, 2]), float(model.site_type[s])]
    self._s_site_geom = torch.as_tensor(sg.copy(), dtype=torch.float32, device=self._device)
    ec = [float(model.opt.gravity[k]) for k in range(3)]
    ec += [float(model.opt.magnetic[k]) for k in range(3)]
    DB = mujoco.mjtDisableBit
    ec += [1.0 if model.opt.disableflags & int(DB.mjDSBL_SPRING) else 0.0,
           1.0 if model.opt.disableflags & int(DB.mjDSBL_DAMPER) else 0.0,
           1.0 if model.opt.disableflags & int(DB.mjDSBL_GRAVITY) else 0.0,
           float(nq)]
    ec += [float(v) for v in np.asarray(model.qpos_spring, dtype=np.float64).reshape(-1)]
    for arr in (np.asarray(model.jnt_stiffness, dtype=np.float64).reshape(-1),
                np.asarray(model.jnt_stiffnesspoly, dtype=np.float64).reshape(nj, 2)[:, 0] if nj else np.zeros(0),
                np.asarray(model.jnt_stiffnesspoly, dtype=np.float64).reshape(nj, 2)[:, 1] if nj else np.zeros(0)):
      ec += [float(v) for v in arr]
    # R07a camera block for CAMPROJECTION (econst tail): [ncam] then per
    # camera [bodyid, lpos(3), lquat(4), res(2), fovy, intrinsic(4),
    # sensorsize(2)]. Kernel offset = 10 + nq + 3*njnt; zero cameras append
    # only the count, bit-identical for existing models.
    ncam = int(model.ncam)
    ec += [float(ncam)]
    for c in range(ncam):
      ec += [float(int(model.cam_bodyid[c]))]
      ec += [float(v) for v in np.asarray(model.cam_pos[c], dtype=np.float64).reshape(-1)]
      ec += [float(v) for v in np.asarray(model.cam_quat[c], dtype=np.float64).reshape(-1)]
      ec += [float(int(model.cam_resolution[c, 0])), float(int(model.cam_resolution[c, 1]))]
      ec += [float(model.cam_fovy[c])]
      ec += [float(v) for v in np.asarray(model.cam_intrinsic[c], dtype=np.float64).reshape(-1)]
      ec += [float(v) for v in np.asarray(model.cam_sensorsize[c], dtype=np.float64).reshape(-1)]
    self._s_econst = torch.as_tensor(np.array(ec, dtype=np.float32), dtype=torch.float32, device=self._device)
    WJ = int(mujoco.mjtWrap.mjWRAP_JOINT)
    ten_lmap = np.zeros((max(nt, 1), max(nq, 1)), dtype=np.float32)
    ten_mmap = np.zeros((max(nt, 1), max(nv, 1)), dtype=np.float32)
    for t in range(nt):
      start, count = int(model.tendon_adr[t]), int(model.tendon_num[t])
      if count <= 0 or int(model.wrap_type[start]) != WJ:
        continue
      ok = True
      for w in range(start, start + count):
        if int(model.wrap_type[w]) != WJ:
          ok = False
          break
        j = int(model.wrap_objid[w])
        if j < 0 or j >= nj or int(model.jnt_type[j]) not in (hinge, slide):
          ok = False
          break
      if not ok:
        continue
      for w in range(start, start + count):
        j = int(model.wrap_objid[w])
        c = float(model.wrap_prm[w])
        ten_lmap[t, int(model.jnt_qposadr[j])] += c
        ten_mmap[t, int(model.jnt_dofadr[j])] += c
    self._s_ten_fix_lmap = torch.as_tensor(ten_lmap.copy(), dtype=torch.float32, device=self._device)
    self._s_ten_fix_mmap = torch.as_tensor(ten_mmap.copy(), dtype=torch.float32, device=self._device)
    JT = int(mujoco.mjtTrn.mjTRN_JOINT)
    JP = int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    TT = int(mujoco.mjtTrn.mjTRN_TENDON)
    act_lmap = np.zeros((max(nu, 1), max(nq, 1)), dtype=np.float32)
    act_mmap = np.zeros((max(nu, 1), max(nv, 1)), dtype=np.float32)
    for a in range(nu):
      trn = int(model.actuator_trntype[a])
      gear = float(np.asarray(model.actuator_gear[a], dtype=np.float64)[0])
      tgt = int(model.actuator_trnid[a, 0])
      if trn in (JT, JP):
        if 0 <= tgt < nj and int(model.jnt_type[tgt]) in (hinge, slide):
          act_lmap[a, int(model.jnt_qposadr[tgt])] = gear
          act_mmap[a, int(model.jnt_dofadr[tgt])] = gear
      elif trn == TT and 0 <= tgt < nt:
        start, count = int(model.tendon_adr[tgt]), int(model.tendon_num[tgt])
        ok = count > 0
        for w in range(start, start + count):
          if int(model.wrap_type[w]) != WJ:
            ok = False
            break
          j = int(model.wrap_objid[w])
          if j < 0 or j >= nj or int(model.jnt_type[j]) not in (hinge, slide):
            ok = False
            break
        if ok:
          for w in range(start, start + count):
            j = int(model.wrap_objid[w])
            c = gear * float(model.wrap_prm[w])
            act_lmap[a, int(model.jnt_qposadr[j])] += c
            act_mmap[a, int(model.jnt_dofadr[j])] += c
    self._s_act_stat_lmap = torch.as_tensor(act_lmap.copy(), dtype=torch.float32, device=self._device)
    self._s_act_stat_mmap = torch.as_tensor(act_mmap.copy(), dtype=torch.float32, device=self._device)
    self._s_state_kernel = self._library.evaluate_state_sensors
    self._s_subtree_kernel = self._library.build_sensor_subtrees
    # Four leading floats preserve 16-byte alignment and carry the stage mask;
    # each body record is 128 bytes, verified in the Metal source ABI.
    self._s_subtree_runtime = torch.zeros(
        self._workspace_sizes["subtree_runtime"], dtype=torch.float32, device=self._device)
    self._s_dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._s_state_out = torch.zeros((self.batch_size, d.nsensordata), dtype=torch.float32, device=self._device)
    self._build_rne_constants(model)

  def run_device(self, qpos, qvel, time, poses, *, stages=(_POS, _VEL), sensordata=None):
    """Update selected sensor stages and return borrowed device sensordata.

    ``poses`` is the device-output mapping from MetalKinematics.run_device.
    ``sensordata`` optionally supplies prior output to preserve unselected
    stages, matching MuJoCo's per-stage in-place updates.
    """
    torch, d = self._torch, self.descriptor
    if any(t.device.type != "mps" for t in (qpos, qvel, time)):
      raise ValueError("qpos, qvel, and time must be MPS tensors")
    for name, tensor in (("qpos", qpos), ("qvel", qvel), ("time", time)):
      if tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous float32 tensor")
    if tuple(qpos.shape) != (self.batch_size, d.nq) or tuple(qvel.shape) != (self.batch_size, d.nv) or tuple(time.shape) != (self.batch_size,):
      raise ValueError("state tensor shapes do not match sensor batch")
    required = ("body_pos", "body_quat", "inertial_pos", "inertial_quat")
    expected_pose_shapes = {
        "body_pos": (self.batch_size, d.nbody, 3),
        "body_quat": (self.batch_size, d.nbody, 4),
        "inertial_pos": (self.batch_size, d.nbody, 3),
        "inertial_quat": (self.batch_size, d.nbody, 4),
        "geom_pos": (self.batch_size, d.ngeom, 3),
        "geom_quat": (self.batch_size, d.ngeom, 4),
        "site_pos": (self.batch_size, d.nsite, 3),
        "site_quat": (self.batch_size, d.nsite, 4),
    }
    for key, shape in expected_pose_shapes.items():
      if key not in poses or poses[key].device.type != "mps":
        raise ValueError(f"poses must contain MPS {key}")
      if tuple(poses[key].shape) != shape or poses[key].dtype != torch.float32 or not poses[key].is_contiguous():
        raise ValueError(f"poses[{key!r}] must be contiguous float32 with shape {shape}")
    cvel = root_com = None
    if self._needs_velocity:
      for key, size in (("cvel", self.batch_size * d.nbody * 6), ("root_com", self.batch_size * d.nbody * 3)):
        if key not in poses or poses[key].device.type != "mps" or poses[key].dtype != torch.float32 or not poses[key].is_contiguous() or poses[key].numel() != size:
          raise ValueError(f"velocity sensors require contiguous float32 MPS poses[{key!r}] with {size} values")
      cvel, root_com = poses["cvel"].reshape(-1), poses["root_com"].reshape(-1)
    else:
      cvel = torch.zeros(1, dtype=torch.float32, device=self._device)
      root_com = cvel
    base = self._output if sensordata is None else sensordata
    if base.device.type != "mps" or tuple(base.shape) != (self.batch_size, d.nsensordata) or base.dtype != torch.float32 or not base.is_contiguous():
      raise ValueError("sensordata must be an MPS tensor matching model output")
    if base is not self._output:
      self._output.copy_(base)
    stages = tuple(int(s) for s in stages)
    if any(s not in (_POS, _VEL) for s in stages) or len(set(stages)) != len(stages):
      raise ValueError("stages must contain unique position/velocity stage ids")
    stage_mask = len(stages) and sum(1 << s for s in stages) or 0
    if stage_mask < 0 or stage_mask > np.iinfo(np.int32).max:
      raise ValueError("invalid sensor stage mask")
    self._stage_mask.fill_(stage_mask)
    self._kernel(qpos, qvel, time, poses["body_pos"], poses["body_quat"], poses["inertial_pos"], poses["inertial_quat"], poses["geom_pos"], poses["geom_quat"], poses["site_pos"], poses["site_quat"], cvel, root_com, *(self._meta[n] for n in _KERNEL_METADATA), self._dims, self._output, self._stage_mask, threads=(self.batch_size * max(d.nsensor, 1),), group_size=(1,))
    return self._output

  def run_state_device(self, qpos, qvel, poses, *, mass_matrix=None,
                       ten_spa_len=None, ten_spa_vel=None,
                       act_dyn_len=None, act_dyn_vel=None, act_kind=0,
                       out=None, stages=(_POS, _VEL)):
    """Evaluate tendon/actuator/limit/subtree/insidesite/energy/magnetometer
    families on MPS and return borrowed device sensordata.

    ``ten_spa_*`` are full-tendon spatial kinematics views (zero rows for
    fixed tendons); ``act_dyn_*`` are general-path kinematics views. Either
    may be None when the model has no spatial tendons / general actuators
    (dummy storage is bound instead). ``mass_matrix`` is required only when
    kinetic-energy sensors are present.
    """
    torch, d = self._torch, self.descriptor
    b = self.batch_size
    for name, tensor, shape in (
        ("qpos", qpos, (b, d.nq)), ("qvel", qvel, (b, d.nv))):
      if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "mps"
          or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape
          or not tensor.is_contiguous()):
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    for key, shape in (
        ("body_pos", (b, d.nbody, 3)), ("body_quat", (b, d.nbody, 4)),
        ("inertial_pos", (b, d.nbody, 3)), ("inertial_quat", (b, d.nbody, 4)),
        ("site_pos", (b, d.nsite, 3)), ("site_quat", (b, d.nsite, 4)),
        ("cvel", (b, d.nbody, 6)), ("root_com", (b, d.nbody, 3))):
      v = poses.get(key, None)
      if (not isinstance(v, torch.Tensor) or v.device.type != "mps"
          or v.dtype != torch.float32 or tuple(v.shape) != shape
          or not v.is_contiguous()):
        raise ValueError(f"poses[{key!r}] must be contiguous float32 MPS with shape {shape}")
    nq, nv, nt, nu = max(d.nq, 1), max(d.nv, 1), max(d.ntendon, 1), max(d.nu, 1)
    def _view(buf, shape, name):
      if buf is None:
        return self._s_dummy
      if (not isinstance(buf, torch.Tensor) or buf.device.type != "mps"
          or buf.dtype != torch.float32 or tuple(buf.shape) != shape
          or not buf.is_contiguous()):
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
      return buf.reshape(-1)
    ten_l = _view(ten_spa_len, (b, max(d.ntendon, 1)), "ten_spa_len")
    ten_v = _view(ten_spa_vel, (b, max(d.ntendon, 1)), "ten_spa_vel")
    act_l = _view(act_dyn_len, (b, max(d.nu, 1)), "act_dyn_len")
    act_v = _view(act_dyn_vel, (b, max(d.nu, 1)), "act_dyn_vel")
    has_ke = bool(np.any(d.sensor_type == int(mujoco.mjtSensor.mjSENS_E_KINETIC)))
    if has_ke:
      if (mass_matrix is None or not isinstance(mass_matrix, torch.Tensor)
          or mass_matrix.device.type != "mps" or mass_matrix.dtype != torch.float32
          or tuple(mass_matrix.shape) != (b, d.nv, d.nv) or not mass_matrix.is_contiguous()):
        raise ValueError("kinetic-energy sensors require the (batch, nv, nv) mass matrix")
      mm = mass_matrix.reshape(-1)
    else:
      mm = self._s_dummy
    dest = self._s_state_out if out is None else out
    if (dest.device.type != "mps" or tuple(dest.shape) != (b, d.nsensordata)
        or dest.dtype != torch.float32 or not dest.is_contiguous()):
      raise ValueError("out must be contiguous float32 MPS with shape (batch, nsensordata)")
    if dest is not self._s_state_out:
      self._s_state_out.copy_(dest)
      dest = self._s_state_out
    stages = tuple(int(s) for s in stages)
    if any(s not in (_POS, _VEL, _ACC) for s in stages) or len(set(stages)) != len(stages):
      raise ValueError("stages must contain unique stage ids")
    mask = sum(1 << s for s in stages)
    self._stage_mask.fill_(int(mask))
    sdims = torch.tensor(
        [d.nsensor, d.nsensordata, d.nq, d.nv, d.njnt, d.nbody, d.ngeom,
         d.nsite, b, d.disableflags, d.ntendon, d.nu, int(act_kind),
         1 if has_ke else 0],
        dtype=torch.int32, device=self._device)
    self._s_subtree_runtime[0:1].fill_(int(mask))
    self._s_subtree_kernel(
        poses["inertial_pos"].reshape(-1), poses["inertial_quat"].reshape(-1),
        poses["cvel"].reshape(-1), poses["root_com"].reshape(-1),
        self._s_massub.reshape(-1), self._s_body_tree.reshape(-1), sdims,
        self._s_subtree_runtime, threads=(b,), group_size=(1,))
    self._s_state_kernel(
        qpos.reshape(-1) if d.nq else self._s_dummy,
        qvel.reshape(-1) if d.nv else self._s_dummy,
        poses["body_pos"].reshape(-1), poses["inertial_pos"].reshape(-1),
        poses["inertial_quat"].reshape(-1), poses["body_quat"].reshape(-1),
        poses["site_pos"].reshape(-1), poses["site_quat"].reshape(-1),
        poses["cvel"].reshape(-1), poses["root_com"].reshape(-1), mm,
        self._s_ten_fix_lmap.reshape(-1), self._s_ten_fix_mmap.reshape(-1),
        ten_l, ten_v,
        self._s_act_stat_lmap.reshape(-1), self._s_act_stat_mmap.reshape(-1),
        act_l, act_v,
        self._s_meta.reshape(-1),
        poses["geom_pos"].reshape(-1) if d.ngeom else self._s_dummy,
        self._s_jnt_meta.reshape(-1), self._s_jnt_lim.reshape(-1),
        self._s_ten_lim.reshape(-1), self._s_massub.reshape(-1),
        self._s_body_tree.reshape(-1), self._s_site_geom.reshape(-1),
        self._s_econst, sdims, dest.reshape(-1), self._s_subtree_runtime,
        threads=(b * max(d.nsensor, 1),), group_size=(1,))
    return dest

  def _build_rne_constants(self, model):
    """Host constants for the milestone-016 RNE-post/ACC kernels."""
    torch, d = self._torch, self.descriptor
    nb, nj, nv = d.nbody, d.njnt, d.nv
    def itensor(values, shape):
      arr = np.asarray(values, dtype=np.int32).reshape(-1)
      if arr.size == 0:
        arr = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.int32)
      return torch.as_tensor(arr.reshape(shape).copy(), dtype=torch.int32, device=self._device)
    def ftensor(values, shape):
      arr = np.asarray(values, dtype=np.float32).reshape(-1)
      if arr.size == 0:
        arr = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.float32)
      if not np.all(np.isfinite(arr)):
        raise ValueError("RNE constants must be finite float32")
      return torch.as_tensor(arr.reshape(shape).copy(), dtype=torch.float32, device=self._device)
    bjj = np.zeros((max(nb, 1), 4), dtype=np.int32)
    for b in range(nb):
      bjj[b] = [int(model.body_jntadr[b]), int(model.body_jntnum[b]),
                int(model.body_dofadr[b]), int(model.body_dofnum[b])]
    self._rne_body_jnt = itensor(bjj, (max(nb, 1), 4))
    self._rne_dof_jntid = itensor(model.dof_jntid, (max(nv, 1),))
    self._rne_jnt_type = itensor(model.jnt_type, (max(nj, 1),))
    self._rne_jnt_dofadr = itensor(model.jnt_dofadr, (max(nj, 1),))
    self._rne_mass = ftensor(model.body_mass, (max(nb, 1),))
    self._rne_body_tree = itensor(
        np.stack([np.asarray(model.body_parentid).reshape(-1),
                  np.asarray(model.body_rootid).reshape(-1)], axis=1),
        (max(nb, 1), 2))
    self._rne_inertia = ftensor(np.asarray(model.body_inertia).reshape(max(nb, 1), 3),
                                (max(nb, 1), 3))
    self._rne_gravity = ftensor(model.opt.gravity, (3,))
    self._rne_body_weld = itensor(model.body_weldid, (max(nb, 1),))
    self._rne_geom_bodyid = itensor(model.geom_bodyid, (max(int(model.ngeom), 1),))
    self._rne_site_bodyid = itensor(model.site_bodyid, (max(int(model.nsite), 1),))
    neq = int(model.neq)
    eqm = np.zeros((max(neq, 1), 4), dtype=np.int32)
    for e in range(neq):
      eqm[e] = [int(model.eq_type[e]), int(model.eq_objtype[e]),
                int(model.eq_obj1id[e]), int(model.eq_obj2id[e])]
    self._rne_eq_meta = itensor(eqm, (max(neq, 1), 4))
    self._rne_eq_data = ftensor(model.eq_data, (max(neq, 1), 11))
    self._rne_site_lpos = ftensor(model.site_pos, (max(int(model.nsite), 1), 3))
    self._rne_act_trn = itensor(
        np.stack([np.asarray(model.actuator_trntype).reshape(-1),
                  np.asarray(model.actuator_trnid).reshape(-1, 2)[:, 0]], axis=1),
        (max(int(model.nu), 1), 2))
    from pathlib import Path as _Path
    self._rne_lib = torch.mps.compile_shader(
        (_Path(__file__).parent / "shaders" / "sensors_rne.metal").read_text())
    self._rne_assemble = self._rne_lib.assemble_cfrc_ext
    self._rne_post = self._rne_lib.rne_post
    self._rne_acc = self._rne_lib.evaluate_acc_sensors
    b = self.batch_size
    nb = self.descriptor.nbody
    self._rne_cacc = torch.zeros((b, max(nb, 1), 6), dtype=torch.float32, device=self._device)
    self._rne_cfrc = torch.zeros_like(self._rne_cacc)
    self._rne_scom = torch.zeros((b, max(nb, 1), 3), dtype=torch.float32, device=self._device)
    self._rne_ext = torch.zeros_like(self._rne_cacc)
    # Reused sequentially by external-force assembly and the RNE post pass.
    self._rne_com_scratch = torch.zeros(
        (b, max(nb, 1), 12), dtype=torch.float32, device=self._device)
    self._build_spatial_constants(model)

  def _build_spatial_constants(self, model):
    """Host constants for the milestone-016 spatial-query kernels."""
    torch = self._torch
    ng = int(model.ngeom)
    nb = self.descriptor.nbody
    nsite = self.descriptor.nsite
    nmat = int(model.nmat)
    def itensor(values, shape):
      arr = np.asarray(values, dtype=np.int32).reshape(-1)
      if arr.size == 0:
        arr = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.int32)
      return torch.as_tensor(arr.reshape(shape).copy(), dtype=torch.int32, device=self._device)
    def ftensor(values, shape):
      arr = np.asarray(values, dtype=np.float32).reshape(-1)
      if arr.size == 0:
        arr = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.float32)
      if not np.all(np.isfinite(arr)):
        raise ValueError("spatial constants must be finite float32")
      return torch.as_tensor(arr.reshape(shape).copy(), dtype=torch.float32, device=self._device)
    self._sp_geom_type = itensor(model.geom_type, (max(ng, 1),))
    self._sp_geom_size = ftensor(model.geom_size, (max(ng, 1), 3))
    self._sp_geom_bodyid = itensor(model.geom_bodyid, (max(ng, 1),))
    self._sp_geom_matid = itensor(model.geom_matid, (max(ng, 1),))
    self._sp_geom_rgba = ftensor(model.geom_rgba, (max(ng, 1), 4))
    self._sp_nmat = nmat
    self._sp_mat_rgba = ftensor(model.mat_rgba, (max(nmat, 1), 4))
    badr = np.zeros((max(nb, 1), 2), dtype=np.int32)
    for b in range(nb):
      badr[b] = [int(model.body_geomadr[b]), int(model.body_geomnum[b])]
    self._sp_body_geoms = itensor(badr, (max(nb, 1), 2))
    rb = np.asarray(model.geom_rbound, dtype=np.float32).reshape(-1).copy()
    for g in range(ng):
      if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[g])
        if mid >= 0:
          vadr = int(model.mesh_vertadr[mid])
          import numpy as _np
          verts = _np.asarray(model.mesh_vert[vadr:vadr + int(model.mesh_vertnum[mid])],
                              dtype=_np.float64).reshape(-1, 3)
          if len(verts):
            rb[g] = max(float(rb[g]), float(_np.max(_np.linalg.norm(verts, axis=1))))
    self._sp_geom_rbound = ftensor(rb.reshape(max(ng, 1)), (max(ng, 1),))
    self._sp_sdf_maxn = min(int(model.opt.sdf_initpoints), 8)
    # Sensor hull store for rays/geomdist: same layout as the coupled
    # mesh/hfield sections, but built for ALL scene meshes/hfields (rays
    # need occlusion hulls even for contype-0 geoms that never appear in
    # contact pairs). SDF section omitted: SDF spatial queries are rejected.
    from mujoco_metal.coupled_constraints import (
        _HF_MAX_DATA, _HF_MAX_GEOMS, _HF_MAX_N, _MESH_DATA_FLOATS,
        _MESH_MAX_FACES, _MESH_MAX_TOTAL, _MESH_MAX_VERTS,
        mesh_hull_is_convex)
    _MESH_T = int(mujoco.mjtGeom.mjGEOM_MESH)
    _HF_T = int(mujoco.mjtGeom.mjGEOM_HFIELD)
    _mesh_used = sorted(g for g in range(ng) if int(model.geom_type[g]) == _MESH_T)
    _hf_used = sorted(g for g in range(ng) if int(model.geom_type[g]) == _HF_T)
    if _mesh_used or _hf_used:
      if len(_hf_used) > _HF_MAX_GEOMS:
        raise ValueError(f"at most {_HF_MAX_GEOMS} heightfield geoms; found {len(_hf_used)}")
      _hull = np.zeros((_MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS + _HF_MAX_DATA,), dtype=np.float32)
      _hinfo = np.full((max(ng, 1) * 9,), -1, dtype=np.int32)
      _cursor, _ncursor = 0, 3 * _MESH_MAX_TOTAL
      _icursor = 3 * (_MESH_MAX_TOTAL + _MESH_MAX_FACES)
      for g in _mesh_used:
        mid = int(model.geom_dataid[g])
        if mid < 0:
          raise ValueError(f"mesh geom {g} has no asset")
        vnum = int(model.mesh_vertnum[mid])
        if vnum <= 0 or vnum > _MESH_MAX_VERTS:
          raise ValueError(f"mesh geom {g} has {vnum} verts; 011 supports 1..{_MESH_MAX_VERTS}")
        if not mesh_hull_is_convex(model, mid):
          raise ValueError(f"mesh geom {g} is non-convex; 016 geomdist requires convex meshes")
        if _cursor + vnum > _MESH_MAX_TOTAL:
          raise ValueError("total mesh hull verts exceed 011 capacity")
        vadr = int(model.mesh_vertadr[mid])
        verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float32)
        _hull[3 * _cursor:3 * (_cursor + vnum)] = verts.reshape(-1)
        _hinfo[9 * g] = _cursor
        _hinfo[9 * g + 1] = vnum
        _cursor += vnum
        fadr = int(model.mesh_faceadr[mid])
        fnum = int(model.mesh_facenum[mid])
        faces = np.asarray(model.mesh_face[fadr:fadr + fnum], dtype=np.int64)
        if _ncursor + 3 * fnum > 3 * (_MESH_MAX_TOTAL + _MESH_MAX_FACES) or \
            _icursor + 3 * fnum > _MESH_DATA_FLOATS:
          raise ValueError("total mesh faces exceed 011 capacity")
        vd = verts.astype(np.float64)
        interior = vd.mean(axis=0)
        _hinfo[9 * g + 2] = _ncursor // 3
        _hinfo[9 * g + 3] = _icursor // 3
        _hinfo[9 * g + 4] = fnum
        for f in faces:
          i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
          n = np.cross(vd[i1] - vd[i0], vd[i2] - vd[i0])
          nl = float(np.linalg.norm(n))
          if nl < 1e-12:
            n = np.zeros(3)
          else:
            n = n / nl
            if float(n @ (vd[[i0, i1, i2]].mean(axis=0) - interior)) < 0.0:
              n = -n
          _hull[_ncursor:_ncursor + 3] = n.astype(np.float32)
          _hull[_icursor:_icursor + 3] = np.array([i0, i1, i2], dtype=np.float32)
          _ncursor += 3
          _icursor += 3
      _scursor = _MESH_DATA_FLOATS
      _dcursor = _MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS
      for g in _hf_used:
        hid = int(model.geom_dataid[g])
        if hid < 0:
          raise ValueError(f"heightfield geom {g} has no asset")
        nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
        if not (2 <= nrow <= _HF_MAX_N and 2 <= ncol <= _HF_MAX_N):
          raise ValueError(f"heightfield dims {(nrow, ncol)} outside 012 bounds")
        ndata = nrow * ncol
        if _dcursor + ndata > len(_hull):
          raise ValueError("total heightfield data exceeds 012 capacity")
        _hull[_scursor:_scursor + 4] = np.asarray(model.hfield_size[hid], dtype=np.float32)
        adr = int(model.hfield_adr[hid])
        _hull[_dcursor:_dcursor + ndata] = np.asarray(
            model.hfield_data[adr:adr + ndata], dtype=np.float32)
        _hinfo[9 * g + 5] = _dcursor
        _hinfo[9 * g + 6] = nrow
        _hinfo[9 * g + 7] = ncol
        _hinfo[9 * g + 8] = _scursor
        _scursor += 4
        _dcursor += ndata
      self._sp_hull = torch.as_tensor(_hull.copy(), dtype=torch.float32, device=self._device)
      self._sp_hull_info = torch.as_tensor(_hinfo.copy(), dtype=torch.int32, device=self._device)
    else:
      self._sp_hull = torch.zeros(1, dtype=torch.float32, device=self._device)
      self._sp_hull_info = torch.zeros(9, dtype=torch.int32, device=self._device)
    from pathlib import Path as _Path
    _base = _Path(__file__).parent / "shaders"
    from mujoco_metal.coupled_constraints import (
        _COLLISION_SHADER, _CONVEX_SHADER, _SDF_SHADER)
    src = (_COLLISION_SHADER.read_text() + "\n" + _CONVEX_SHADER.read_text()
           + "\n" + _SDF_SHADER.read_text() + "\n"
           + (_base / "sensors_spatial.metal").read_text())
    self._sp_lib = torch.mps.compile_shader(src)
    self._sp_contact = self._sp_lib.evaluate_contact_sensors
    self._sp_rays = self._sp_lib.evaluate_rays
    self._sp_geomdist = self._sp_lib.evaluate_geomdist
    self._sp_tactile = self._sp_lib.evaluate_tactile
    self._build_tactile_constants(model)

  def _build_tactile_constants(self, model):
    """Lower immutable taxel frames and distance octrees, without step readbacks."""
    torch = self._torch
    metadata, frames = [], []
    for i in self._tactile_sensors:
      mesh, geom = int(model.sensor_objid[i]), int(model.sensor_refid[i])
      nvert = int(model.mesh_vertnum[mesh])
      vadr, nadr = int(model.mesh_vertadr[mesh]), int(model.mesh_normaladr[mesh])
      has_frame = int(model.mesh_normalnum[mesh]) == 3 * nvert
      channels = int(model.sensor_dim[i]) // nvert
      verts = np.asarray(model.mesh_vert[vadr:vadr + nvert])
      normals = np.asarray(model.mesh_normal[nadr:nadr + (3*nvert if has_frame else nvert)])
      quat = np.asarray(model.mesh_quat[mesh])
      weld = int(model.body_weldid[int(model.geom_bodyid[geom])])
      for j in range(nvert):
        tangents = np.zeros((2, 3))
        if has_frame:
          tangents[:] = normals[3*j + 1:3*j + 3]
          tangents += 2 * np.cross(quat[1:],
                                   np.cross(quat[1:], tangents) + quat[0]*tangents)
        metadata.append([geom, weld, nvert, int(model.sensor_adr[i])+j,
                         int(has_frame), channels])
        frames.append(np.concatenate([verts[j], tangents.reshape(-1),
                                      [model.sensor_cutoff[i]]]))
    self._tactile_count = len(metadata)
    self._tactile_has_frame = any(row[4] for row in metadata)
    self._tactile_meta = torch.as_tensor(
        np.asarray(metadata or [[0]*6], dtype=np.int32).reshape(-1), device=self._device)
    self._tactile_frames = torch.as_tensor(
        np.asarray(frames or [[0.0]*10], dtype=np.float32).reshape(-1), device=self._device)
    # A mesh without a compiled octree is intentionally skipped by the pinned
    # tactile routine. SDFs and needsdf meshes share its oct_distance helper.
    info = np.full((max(int(model.ngeom), 1), 2), -1, dtype=np.int32)
    blocks, offsets, cursor = [], {}, 0
    for g in range(int(model.ngeom)):
      if int(model.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_MESH),
                                         int(mujoco.mjtGeom.mjGEOM_SDF)):
        continue
      mesh = int(model.geom_dataid[g])
      if mesh < 0 or int(model.mesh_octadr[mesh]) < 0:
        continue
      if mesh not in offsets:
        adr, count = int(model.mesh_octadr[mesh]), int(model.mesh_octnum[mesh])
        if count <= 0 or count >= 2**24:
          raise ValueError("tactile octree must have a positive float32-exact node count")
        block = np.concatenate([
            np.asarray(model.oct_child[adr:adr+count]).reshape(-1),
            np.asarray(model.oct_aabb[adr:adr+count]).reshape(-1),
            np.asarray(model.oct_coeff[adr:adr+count]).reshape(-1)]).astype(np.float32)
        if block.size != 22*count or not np.all(np.isfinite(block)):
          raise ValueError("invalid compiled tactile octree layout")
        offsets[mesh] = cursor, count
        cursor += block.size
        blocks.append(block)
      info[g] = offsets[mesh]
    self._tactile_oct_info = torch.as_tensor(info.reshape(-1), device=self._device)
    self._tactile_oct = torch.as_tensor(
        np.concatenate(blocks) if blocks else np.zeros(1, dtype=np.float32),
        device=self._device)
    self._tactile_dims = torch.tensor(
        [self.batch_size, int(model.ngeom), int(model.nbody), int(model.nsensordata),
         0, self._tactile_count, int(bool(int(model.opt.disableflags)
             & int(mujoco.mjtDisableBit.mjDSBL_SENSOR)))],
        dtype=torch.int32, device=self._device)

  def _sp_validate_poses(self, poses, need_geom_quat=True):
    torch, d, b = self._torch, self.descriptor, self.batch_size
    for key, shape in (
        ("site_pos", (b, d.nsite, 3)), ("site_quat", (b, d.nsite, 4)),
        ("geom_pos", (b, d.ngeom, 3))):
      v = poses.get(key, None)
      if (not isinstance(v, torch.Tensor) or v.device.type != "mps"
          or v.dtype != torch.float32 or tuple(v.shape) != shape
          or not v.is_contiguous()):
        raise ValueError(f"poses[{key!r}] must be contiguous float32 MPS with shape {shape}")
    if need_geom_quat:
      v = poses.get("geom_quat", None)
      if (not isinstance(v, torch.Tensor) or v.device.type != "mps"
          or v.dtype != torch.float32 or tuple(v.shape) != (b, d.ngeom, 4)
          or not v.is_contiguous()):
        raise ValueError("poses['geom_quat'] must be contiguous float32 MPS")

  def run_contact_device(self, poses, contact, out=None):
    """Evaluate CONTACT sensors into ``out`` (borrowed, merged)."""
    torch, d, b = self._torch, self.descriptor, self.batch_size
    self._sp_validate_poses(poses, need_geom_quat=bool(getattr(self, "_tactile_sensors", None)))
    nc = 0 if contact is None else int(contact["frame"].shape[1])
    dummy = self._s_dummy
    if contact is None:
      cfr = cfo = crow = cmu = dummy
      cpk = torch.zeros(3, dtype=torch.int32, device=self._device)
      slot = torch.zeros(1, dtype=torch.int32, device=self._device)
      live = torch.zeros(1, dtype=torch.int32, device=self._device)
      pge = torch.zeros(2, dtype=torch.int32, device=self._device)
    else:
      cfr, cfo = contact["frame"].reshape(-1), contact["force"].reshape(-1)
      crow, cpk = contact["row"].reshape(-1), contact["packed"].reshape(-1)
      cmu = contact["mu"].reshape(-1)
      slot, live = contact["slot_pair"].reshape(-1), contact["pair_live"].reshape(-1)
      pge = contact["pair_geoms"].reshape(-1)
    dest = self._s_state_out if out is None else out
    if dest.device.type != "mps" or tuple(dest.shape) != (b, d.nsensordata) \
        or dest.dtype != torch.float32 or not dest.is_contiguous():
      raise ValueError("out must be contiguous float32 MPS with shape (batch, nsensordata)")
    if dest is not self._s_state_out:
      self._s_state_out.copy_(dest)
      dest = self._s_state_out
    self._stage_mask.fill_(1 << int(mujoco.mjtStage.mjSTAGE_ACC))
    cdims = torch.tensor(
        [b, d.nsensor, d.nsensordata, nc, d.nsite, d.nbody, d.disableflags],
        dtype=torch.int32, device=self._device)
    self._sp_contact(
        cfr, cfo, crow, cpk, cmu, slot, live, pge,
        self._rne_geom_bodyid, self._rne_site_bodyid, self._s_body_tree.reshape(-1),
        poses["site_pos"].reshape(-1), poses["site_quat"].reshape(-1),
        self._s_site_geom.reshape(-1),
        self._s_meta.reshape(-1),
        self._s_intprm(),
        dest.reshape(-1), self._stage_mask, cdims,
        threads=(b * max(d.nsensor, 1),), group_size=(1,))
    if getattr(self, "_tactile_sensors", None):
      self._evaluate_tactile(poses, contact, dest)
    return dest

  def _evaluate_tactile(self, poses, contact, dest):
    """Evaluate taxels with device-resident contact eligibility and reductions."""
    if not self._tactile_count:
      return
    torch, d, batch = self._torch, self.descriptor, self.batch_size
    nc = 0 if contact is None else int(contact["frame"].shape[1])
    self._tactile_dims[4] = nc
    dummy = self._s_dummy
    if contact is None or nc == 0:
      contact_frames, slots, pairs = dummy, self._tactile_meta, self._tactile_meta
    else:
      # Tactile uses all detected contacts, including margin/gap contacts
      # without solver rows. An unused slot has a zero contact frame.
      contact_frames = contact["frame"].contiguous().reshape(-1)
      slots = contact["slot_pair"].reshape(-1)
      pairs = contact["pair_geoms"].reshape(-1)
    cvel, com = poses.get("cvel"), poses.get("root_com")
    if cvel is None or com is None:
      if self._tactile_has_frame:
        raise ValueError("framed tactile sensors require stage cvel and root_com")
      cvel, com = self._rne_cacc, self._rne_scom
    self._sp_tactile(
        poses["geom_pos"].reshape(-1), poses["geom_quat"].reshape(-1),
        cvel.reshape(-1), com.reshape(-1), contact_frames, slots, pairs,
        self._rne_geom_bodyid, self._rne_body_weld, self._rne_body_tree.reshape(-1),
        self._sp_geom_type, self._sp_geom_size.reshape(-1),
        self._tactile_meta, self._tactile_frames, self._tactile_oct_info,
        self._tactile_oct, dest.reshape(-1), self._tactile_dims,
        threads=(batch*self._tactile_count,), group_size=(1,))

  def _s_intprm(self):
    import numpy as _np
    return self._torch.as_tensor(
        _np.asarray(self.descriptor.sensor_intprm, dtype=_np.int32).reshape(-1).copy(),
        dtype=self._torch.int32, device=self._device)

  def run_rays_device(self, poses, hull, hull_info, out=None):
    """Evaluate site rangefinders into ``out`` (borrowed, merged)."""
    torch, d, b = self._torch, self.descriptor, self.batch_size
    self._sp_validate_poses(poses, need_geom_quat=True)
    dest = self._s_state_out if out is None else out
    if dest.device.type != "mps" or tuple(dest.shape) != (b, d.nsensordata) \
        or dest.dtype != torch.float32 or not dest.is_contiguous():
      raise ValueError("out must be contiguous float32 MPS with shape (batch, nsensordata)")
    if dest is not self._s_state_out:
      self._s_state_out.copy_(dest)
      dest = self._s_state_out
    self._stage_mask.fill_(1 << int(mujoco.mjtStage.mjSTAGE_POS))
    rdims = torch.tensor(
        [b, d.nsensor, d.nsensordata, d.ngeom, d.nsite, self._sp_nmat,
         d.disableflags, 0],
        dtype=torch.int32, device=self._device)
    self._sp_rays(
        poses["site_pos"].reshape(-1), poses["site_quat"].reshape(-1),
        poses["geom_pos"].reshape(-1), poses["geom_quat"].reshape(-1),
        self._sp_geom_type.reshape(-1), self._sp_geom_size.reshape(-1),
        self._rne_geom_bodyid,
        self._sp_geom_matid.reshape(-1), self._sp_geom_rgba.reshape(-1),
        self._sp_mat_rgba.reshape(-1), hull.reshape(-1), hull_info.reshape(-1),
        self._rne_site_bodyid,
        self._s_meta.reshape(-1), self._s_intprm(),
        dest.reshape(-1), self._stage_mask, rdims,
        threads=(b * max(d.nsensor, 1),), group_size=(1,))
    return dest

  def run_geomdist_device(self, poses, hull, hull_info, out=None):
    """Evaluate geom-distance witnesses into ``out`` (borrowed, merged)."""
    torch, d, b = self._torch, self.descriptor, self.batch_size
    for key, shape in (("geom_pos", (b, d.ngeom, 3)),
                       ("geom_quat", (b, d.ngeom, 4))):
      v = poses.get(key, None)
      if (not isinstance(v, torch.Tensor) or v.device.type != "mps"
          or v.dtype != torch.float32 or tuple(v.shape) != shape
          or not v.is_contiguous()):
        raise ValueError(f"poses[{key!r}] must be contiguous float32 MPS with shape {shape}")
    dest = self._s_state_out if out is None else out
    if dest.device.type != "mps" or tuple(dest.shape) != (b, d.nsensordata) \
        or dest.dtype != torch.float32 or not dest.is_contiguous():
      raise ValueError("out must be contiguous float32 MPS with shape (batch, nsensordata)")
    if dest is not self._s_state_out:
      self._s_state_out.copy_(dest)
      dest = self._s_state_out
    self._stage_mask.fill_(1 << int(mujoco.mjtStage.mjSTAGE_POS))
    gdims = torch.tensor(
        [b, d.nsensor, d.nsensordata, d.ngeom, d.nbody, d.disableflags,
         0, int(self._sp_sdf_maxn), 0],
        dtype=torch.int32, device=self._device)
    self._sp_geomdist(
        poses["geom_pos"].reshape(-1), poses["geom_quat"].reshape(-1),
        self._sp_geom_type.reshape(-1), self._sp_geom_size.reshape(-1),
        self._sp_geom_rbound.reshape(-1),
        self._sp_body_geoms.reshape(-1),
        hull.reshape(-1), hull_info.reshape(-1),
        self._s_meta.reshape(-1),
        dest.reshape(-1), self._stage_mask, gdims,
        threads=(b * max(d.nsensor, 1),), group_size=(1,))
    return dest

  def run_acc_device(self, qpos, qvel, qacc, poses, *, xfrc=None,
                     contact=None, eq_rowadr=None, jnt_map=None,
                     ten_map=None, slot_pair=None, lam_raw=None, lam_nr=0,
                     lam_stride=0, act_force=None, qfrc_act=None, out=None):
    """Evaluate ACC force families on MPS; returns borrowed device sensordata.

    Runs cfrc_ext assembly, the RNE-post pass, then the ACC sensor kernel.
    ``contact`` is a dict of coupled contact views (frame/force/row/packed/
    mu/pair_geoms/pair_offset/pair_live) or None; ``eq_rowadr`` per-equality
    first rows or None; ``jnt_map``/``ten_map`` candidate rows or None;
    ``act_force``/``qfrc_act`` per-actuator/dof forces or None.
    """
    torch, d = self._torch, self.descriptor
    b = self.batch_size
    nb = d.nbody
    for name, tensor, shape in (
        ("qpos", qpos, (b, d.nq)), ("qvel", qvel, (b, d.nv)),
        ("qacc", qacc, (b, d.nv))):
      if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "mps"
          or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape
          or not tensor.is_contiguous()):
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    for key, shape in (
        ("body_pos", (b, nb, 3)), ("body_quat", (b, nb, 4)),
        ("inertial_pos", (b, nb, 3)), ("inertial_quat", (b, nb, 4)),
        ("site_pos", (b, d.nsite, 3)), ("site_quat", (b, d.nsite, 4)),
        ("geom_pos", (b, d.ngeom, 3)),
        ("cvel", (b, nb, 6)), ("root_com", (b, nb, 3))):
      v = poses.get(key, None)
      if (not isinstance(v, torch.Tensor) or v.device.type != "mps"
          or v.dtype != torch.float32 or tuple(v.shape) != shape
          or not v.is_contiguous()):
        raise ValueError(f"poses[{key!r}] must be contiguous float32 MPS with shape {shape}")
    for key, shape in (
        ("joint_anchor", (b, d.njnt, 3)), ("joint_axis", (b, d.njnt, 3))):
      v = poses.get(key, None)
      if (not isinstance(v, torch.Tensor) or v.device.type != "mps"
          or v.dtype != torch.float32 or tuple(v.shape) != shape
          or not v.is_contiguous()):
        raise ValueError(f"poses[{key!r}] must be contiguous float32 MPS with shape {shape}")
    nv, nj, nt, nu = d.nv, d.njnt, d.ntendon, d.nu
    nc = 0 if contact is None else int(contact["frame"].shape[1])
    dummy = self._s_dummy
    XF = xfrc.reshape(-1) if xfrc is not None else dummy
    has_x = 0 if xfrc is None else 1
    if contact is None:
      cfr = cfo = crow = cmu = dummy
      cpk = torch.zeros(3, dtype=torch.int32, device=self._device)
      pge = torch.zeros(2, dtype=torch.int32, device=self._device)
      pof = torch.zeros(2, dtype=torch.int32, device=self._device)
      plv = torch.zeros(1, dtype=torch.int32, device=self._device)
      npairs = 0
    else:
      cfr, cfo = contact["frame"].reshape(-1), contact["force"].reshape(-1)
      # Active-flag view may be strided upstream; materialize the documented
      # contiguous per-world row-major contract before flattening.
      crow, cpk = contact["row"].contiguous().reshape(-1), contact["packed"].reshape(-1)
      cmu, pge = contact["mu"].reshape(-1), contact["pair_geoms"].reshape(-1)
      pof = contact["pair_offset"].reshape(-1)
      plv = contact["pair_live"].reshape(-1)
      npairs = int(contact.get("npairs", 0))
    bjn = self._rne_body_jnt[:, 1].contiguous().reshape(-1)
    eqr = eq_rowadr.reshape(-1) if eq_rowadr is not None else torch.zeros(1, dtype=torch.int32, device=self._device)
    neq_eff = int(eqr.numel()) if eq_rowadr is not None else 0
    jm = jnt_map.reshape(-1) if jnt_map is not None else torch.full((max(nj, 1) * 3,), -1, dtype=torch.int32, device=self._device)
    tm = ten_map.reshape(-1) if ten_map is not None else torch.full((max(nt, 1) * 2,), -1, dtype=torch.int32, device=self._device)
    lam = lam_raw.reshape(-1) if lam_raw is not None else dummy
    af = act_force.reshape(-1) if act_force is not None else dummy
    qa = qfrc_act.reshape(-1) if qfrc_act is not None else dummy
    has_act = 0 if act_force is None else 1
    adims = torch.tensor(
        [b, nb, neq_eff, nc, npairs, has_x, lam_nr, lam_stride],
        dtype=torch.int32, device=self._device)
    self._rne_assemble(
        XF, cfr, cfo, crow, cpk, cmu, pge, pof,
        self._rne_geom_bodyid, bjn,
        self._rne_eq_meta.reshape(-1), eqr,
        self._rne_eq_data.reshape(-1), self._rne_site_lpos.reshape(-1),
        self._rne_site_bodyid,
        lam, poses["body_pos"].reshape(-1), poses["body_quat"].reshape(-1),
        poses["inertial_pos"].reshape(-1),
        self._rne_mass, self._rne_body_tree.reshape(-1),
        self._rne_ext.reshape(-1), adims, self._rne_com_scratch.reshape(-1),
        threads=(b,), group_size=(1,))
    grav_off = 1 if d.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY) else 0
    rdims = torch.tensor(
        [b, nb, nv, nj, grav_off],
        dtype=torch.int32, device=self._device)
    # Root-com-per-body for the ACC kernel (written by rne_post).
    self._rne_post(
        qvel.reshape(-1) if nv else dummy,
        qacc.reshape(-1) if nv else dummy,
        poses["cvel"].reshape(-1),
        self._rne_mass, self._rne_inertia.reshape(-1),
        self._rne_body_tree.reshape(-1), self._rne_body_jnt.reshape(-1),
        self._rne_jnt_type if nj else torch.zeros(1, dtype=torch.int32, device=self._device),
        self._rne_jnt_dofadr if nj else torch.zeros(1, dtype=torch.int32, device=self._device),
        poses["joint_anchor"].reshape(-1) if nj else dummy,
        poses["joint_axis"].reshape(-1) if nj else dummy,
        poses["inertial_pos"].reshape(-1), poses["inertial_quat"].reshape(-1),
        poses["body_quat"].reshape(-1),
        self._rne_ext.reshape(-1),
        self._rne_cacc.reshape(-1), self._rne_cfrc.reshape(-1),
        self._rne_scom.reshape(-1),
        rdims, self._rne_gravity, self._rne_com_scratch.reshape(-1),
        threads=(b,), group_size=(1,))
    dest = self._s_state_out if out is None else out
    if dest.device.type != "mps" or tuple(dest.shape) != (b, d.nsensordata) \
        or dest.dtype != torch.float32 or not dest.is_contiguous():
      raise ValueError("out must be contiguous float32 MPS with shape (batch, nsensordata)")
    if dest is not self._s_state_out:
      self._s_state_out.copy_(dest)
      dest = self._s_state_out
    self._stage_mask.fill_(1 << int(mujoco.mjtStage.mjSTAGE_ACC))
    kdims = torch.tensor(
        [b, d.nsensor, d.nsensordata, lam_nr, lam_stride, nc, nu, nt, nv,
         nj, nb, has_act, d.nsite, d.ngeom],
        dtype=torch.int32, device=self._device)
    self._rne_acc(
        self._rne_cacc.reshape(-1), self._rne_cfrc.reshape(-1),
        self._rne_scom.reshape(-1), af, qa,
        self._rne_act_trn.reshape(-1), jm, tm, lam,
        poses["site_pos"].reshape(-1), poses["site_quat"].reshape(-1),
        poses["body_pos"].reshape(-1), poses["inertial_pos"].reshape(-1),
        poses["geom_pos"].reshape(-1) if d.ngeom else dummy,
        poses["cvel"].reshape(-1), poses["root_com"].reshape(-1),
        cfr, cfo, crow, cpk, cmu,
        slot_pair.reshape(-1) if slot_pair is not None else torch.zeros(1, dtype=torch.int32, device=self._device),
        plv,
        self._s_site_geom.reshape(-1),
        self._s_meta.reshape(-1),
        self._rne_geom_bodyid, self._rne_site_bodyid, self._rne_body_weld,
        dest.reshape(-1), self._stage_mask, kdims,
        threads=(b * max(d.nsensor, 1),), group_size=(1,))
    return dest
