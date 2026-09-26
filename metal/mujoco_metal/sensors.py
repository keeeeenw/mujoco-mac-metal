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
}
_POS = int(mujoco.mjtStage.mjSTAGE_POS)
_VEL = int(mujoco.mjtStage.mjSTAGE_VEL)
_DISABLE_SENSOR = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)


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
      "jnt_type": (np.int32, (nj,)),
      "jnt_qposadr": (np.int32, (nj,)),
      "jnt_dofadr": (np.int32, (nj,)),
      "body_iquat": (np.float32, (nb, 4)),
      "body_rootid": (np.int32, (nb,)),
      "geom_bodyid": (np.int32, (ng,)),
      "geom_pos": (np.float32, (ng, 3)),
      "geom_quat": (np.float32, (ng, 4)),
      "site_bodyid": (np.int32, (nsite,)),
      "site_pos": (np.float32, (nsite, 3)),
      "site_quat": (np.float32, (nsite, 4)),
  }
  arrays = {name: _vec(model, name, dtype, shape) for name, (dtype, shape) in names.items()}
  stages = {_POS, _VEL}
  valid_objtypes = {
      int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY),
      int(mujoco.mjtObj.mjOBJ_GEOM), int(mujoco.mjtObj.mjOBJ_SITE),
  }
  sensor = arrays["sensor_type"]
  for i, typ in enumerate(sensor):
    typ = int(typ)
    if typ not in _SUPPORTED:
      raise ValueError(f"sensor {i}: unsupported MuJoCo sensor type {typ}")
    stage = int(arrays["sensor_needstage"][i])
    expected_stage = _VEL if typ in (
        int(mujoco.mjtSensor.mjSENS_JOINTVEL),
        int(mujoco.mjtSensor.mjSENS_BALLANGVEL),
        int(mujoco.mjtSensor.mjSENS_FRAMELINVEL),
        int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
        int(mujoco.mjtSensor.mjSENS_GYRO),
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ) else _POS
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
    if int(arrays["sensor_datatype"][i]) in (int(mujoco.mjtDataType.mjDATATYPE_AXIS), int(mujoco.mjtDataType.mjDATATYPE_QUATERNION)) and float(arrays["sensor_cutoff"][i]) > 0:
      # MuJoCo cutoff is ignored for these normalized data types.
      pass
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


def _quat_mul(a, b):
  return np.array([a[0]*b[0] - np.dot(a[1:], b[1:]), *(a[0]*b[1:] + b[0]*a[1:] + np.cross(a[1:], b[1:]))])


def _quat_rot(q, v):
  return v + 2 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


def _unit(q, name):
  norm = np.linalg.norm(q)
  if not np.isfinite(norm) or norm <= 0:
    raise ValueError(f"{name} must be a nonzero quaternion")
  return q / norm


def sensor_oracle(model, qpos, qvel, time, poses, sensordata=None, stages=(_POS, _VEL)):
  """Float64 CPU reference for the supported stage; accepts FK pose arrays."""
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
            ref_body = _object_body(desc, ref_type, refid)
            ref_vel = _object_velocity(desc, poses, w, ref_type, refid, ref_pos)
            relative = np.concatenate((obj_vel[:3] - ref_vel[:3], obj_vel[3:] - ref_vel[3:] - np.cross(obj_pos-ref_pos, ref_vel[:3])))
            relative[:3] = _quat_rot(np.array([ref_quat[0], *(-ref_quat[1:])]), relative[:3])
            relative[3:] = _quat_rot(np.array([ref_quat[0], *(-ref_quat[1:])]), relative[3:])
            del ref_body
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
  cvel = np.asarray(poses["cvel"])[world, body]
  root_com = np.asarray(poses["root_com"])[world, root]
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
    for name in ("sensor_type", "sensor_datatype", "sensor_needstage", "sensor_objtype", "sensor_objid", "sensor_reftype", "sensor_refid", "sensor_dim", "sensor_adr", "sensor_cutoff", "jnt_type", "jnt_qposadr", "jnt_dofadr", "body_iquat", "geom_bodyid", "geom_pos", "geom_quat", "site_bodyid", "site_pos", "site_quat"):
      arr = getattr(d, name)
      self._meta[name] = torch.as_tensor(np.array(arr, copy=True), device=self._device)
    self._dims = torch.tensor([d.nsensor, d.nsensordata, d.nq, d.nv, d.njnt, d.nbody, d.ngeom, d.nsite, batch_size, d.disableflags], dtype=torch.int32, device=self._device)
    self._stage_mask = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._output = torch.zeros((batch_size, d.nsensordata), dtype=torch.float32, device=self._device)

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
    self._kernel(qpos, qvel, time, poses["body_pos"], poses["body_quat"], poses["inertial_pos"], poses["inertial_quat"], poses["geom_pos"], poses["geom_quat"], poses["site_pos"], poses["site_quat"], *(self._meta[n] for n in ("sensor_type", "sensor_datatype", "sensor_needstage", "sensor_objtype", "sensor_objid", "sensor_reftype", "sensor_refid", "sensor_dim", "sensor_adr", "sensor_cutoff", "jnt_qposadr", "jnt_dofadr")), self._dims, self._output, self._stage_mask, threads=(self.batch_size * max(d.nsensor, 1),), group_size=(1,))
    return self._output
