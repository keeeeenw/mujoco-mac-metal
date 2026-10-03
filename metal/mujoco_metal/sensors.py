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
    # Milestone 016 families (CAMPROJECTION/TACTILE/PLUGIN/USER stay rejected).
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
}
_DEFERRED_TO_019 = {
    int(mujoco.mjtSensor.mjSENS_CAMPROJECTION),
    int(mujoco.mjtSensor.mjSENS_TACTILE),
    int(mujoco.mjtSensor.mjSENS_PLUGIN),
    int(mujoco.mjtSensor.mjSENS_USER),
}
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
  jnt_type: np.ndarray
  jnt_qposadr: np.ndarray
  jnt_dofadr: np.ndarray
  body_iquat: np.ndarray
  body_rootid: np.ndarray
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
                int(S.mjSENS_FRAMEANGACC), int(S.mjSENS_CONTACT)}
  valid_objtypes = {
      int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY),
      int(mujoco.mjtObj.mjOBJ_GEOM), int(mujoco.mjtObj.mjOBJ_SITE),
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
  _STATE_TYPES = {int(getattr(mujoco.mjtSensor, n)) for n in (
      "mjSENS_TENDONPOS", "mjSENS_TENDONVEL", "mjSENS_ACTUATORPOS",
      "mjSENS_ACTUATORVEL", "mjSENS_JOINTLIMITPOS", "mjSENS_JOINTLIMITVEL",
      "mjSENS_TENDONLIMITPOS", "mjSENS_TENDONLIMITVEL",
      "mjSENS_SUBTREECOM", "mjSENS_SUBTREELINVEL", "mjSENS_SUBTREEANGMOM",
      "mjSENS_INSIDESITE", "mjSENS_E_POTENTIAL", "mjSENS_E_KINETIC",
      "mjSENS_MAGNETOMETER")}
  if nb > 64 and bool(np.any(np.isin(sensor, list(_STATE_TYPES)))):
    raise ValueError("state-family sensors bound nbody to 64")
  # Stateful timing features change only on selected calls and require history.
  if np.any(np.asarray(model.sensor_delay) != 0) or np.any(np.asarray(model.sensor_interval) != 0) or np.any(np.asarray(model.sensor_noise) != 0) or np.any(np.asarray(model.sensor_history) != 0):
    raise ValueError("sensor delay, interval, noise, and history are unsupported")
  return SensorDescriptor(
      model.nsensor, model.nsensordata, model.nq, model.nv, nj, nb, ng, nsite,
      **arrays, disableflags=int(model.opt.disableflags),
  )


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


class SensorProgram:
  """Batched MSL sensor evaluator. Constructor is the explicit MPS boundary."""

  def __init__(self, model, batch_size=1):
    self.descriptor = lower_sensors(model)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    if batch_size > (1 << 31) - 1 or batch_size * max(self.descriptor.nsensor, 1) > (1 << 31) - 1:
      raise ValueError("sensor dispatch exceeds the Metal int32 thread limit")
    if batch_size * self.descriptor.nsensordata > (1 << 32):
      raise ValueError("sensor output exceeds the Metal uint32 index capacity")
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
    self._s_dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._s_state_out = torch.zeros((self.batch_size, d.nsensordata), dtype=torch.float32, device=self._device)

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
        self._s_econst, sdims, dest.reshape(-1), self._stage_mask,
        threads=(b * max(d.nsensor, 1),), group_size=(1,))
    return dest
