# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned MuJoCo sensor sleep-state policy and fixed-shape MPS mask."""

from pathlib import Path

import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "sensor_sleep.metal"


def sensor_sleep_workspace_sizes(model, batch_size, *, validate=True):
  """Exact fixed-shape sensor sleep tensors, checked before device import."""
  from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
  if (isinstance(batch_size, (bool, np.bool_))
      or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0):
    raise ValueError("batch_size must be a positive integer")
  b, nsensor, ntree = (int(batch_size), int(model.nsensor), int(model.ntree))
  sizes = {
      "sensor_sleep.treeids": nsensor * 4,
      "sensor_sleep.counts": nsensor * 2,
      "sensor_sleep.always": nsensor,
      "sensor_sleep.sensor_refs": nsensor * 2,
      "sensor_sleep.output": b * nsensor,
      "sensor_sleep.dims": 3,
  }
  if validate:
    _validate_workspace_index_capacity(
        b, sizes, {"nsensor": nsensor, "ntree": ntree})
    for name, elements in sizes.items():
      if elements > (1 << 31) - 1:
        raise ValueError(
            f"{name} exceeds the Metal int32 address range ({elements})")
  return sizes


def _object_trees(model, objtype, objid, mujoco):
  """Resolve one sleepState object into at most two dynamic tree IDs.

  Return ``None`` for MuJoCo UNKNOWN, ``False`` for an unsupported family
  (which is kept awake conservatively), or a tuple for a known object. An
  empty tuple is a static object. Sentinel tuples encode known always-awake
  ``(-998,)``, known always-asleep ``(-999,)``, or a sensor reference
  ``(-997, sensor_id)``.
  """
  O = mujoco.mjtObj
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  typ, idx = int(objtype), int(objid)
  if typ == int(O.mjOBJ_UNKNOWN):
    return None

  body = -1
  if typ in (int(O.mjOBJ_BODY), int(O.mjOBJ_XBODY)):
    if 0 <= idx < int(model.nbody):
      body = idx
  elif typ == int(O.mjOBJ_JOINT):
    ids = np.asarray(model.jnt_bodyid, dtype=np.int32)
    if 0 <= idx < ids.size:
      body = int(ids[idx])
  elif typ == int(O.mjOBJ_DOF):
    ids = np.asarray(model.dof_bodyid, dtype=np.int32)
    if 0 <= idx < ids.size:
      body = int(ids[idx])
  elif typ == int(O.mjOBJ_SITE):
    ids = np.asarray(model.site_bodyid, dtype=np.int32)
    if 0 <= idx < ids.size:
      body = int(ids[idx])
  elif typ == int(O.mjOBJ_GEOM):
    ids = np.asarray(model.geom_bodyid, dtype=np.int32)
    if 0 <= idx < ids.size:
      body = int(ids[idx])
  elif typ == int(O.mjOBJ_CAMERA):
    ids = np.asarray(model.cam_bodyid, dtype=np.int32)
    if 0 <= idx < ids.size:
      body = int(ids[idx])
  elif typ == int(O.mjOBJ_LIGHT):
    ids = np.asarray(model.light_bodyid, dtype=np.int32)
    if 0 <= idx < ids.size:
      body = int(ids[idx])
  elif typ == int(O.mjOBJ_TENDON):
    if not 0 <= idx < int(model.ntendon):
      return False
    count = int(model.tendon_treenum[idx])
    if count > 2:
      return False
    tree_ids = np.asarray(model.tendon_treeid, dtype=np.int32).reshape(-1, 2)[idx]
    return tuple(sorted({int(t) for t in tree_ids[:count] if int(t) >= 0}))
  elif typ == int(O.mjOBJ_FLEX):
    if not 0 <= idx < int(model.nflex):
      return False
    if bool(model.flex_interp[idx]):
      adr, count = int(model.flex_nodeadr[idx]), int(model.flex_nodenum[idx])
      bodies = np.asarray(model.flex_nodebodyid, dtype=np.int32)[adr:adr + count]
    else:
      adr, count = int(model.flex_vertadr[idx]), int(model.flex_vertnum[idx])
      bodies = np.asarray(model.flex_vertbodyid, dtype=np.int32)[adr:adr + count]
    # Pinned mj_sleepState(mjOBJ_FLEX) checks the first dynamic node/vertex.
    for body_id in bodies:
      tree = int(body_tree[int(body_id)])
      if tree >= 0:
        return (tree,)
    return ()
  elif typ == int(O.mjOBJ_EQUALITY):
    if not 0 <= idx < int(model.neq):
      return False
    E = mujoco.mjtEq
    eqtype = int(model.eq_type[idx])
    if eqtype in (int(E.mjEQ_CONNECT), int(E.mjEQ_WELD)):
      target_type = int(model.eq_objtype[idx])
    elif eqtype == int(E.mjEQ_JOINT):
      target_type = int(O.mjOBJ_JOINT)
    elif eqtype == int(E.mjEQ_TENDON):
      target_type = int(O.mjOBJ_TENDON)
    elif eqtype in (int(E.mjEQ_FLEX), int(E.mjEQ_FLEXVERT),
                    int(E.mjEQ_FLEXSTRAIN)):
      target_type = int(O.mjOBJ_FLEX)
    else:
      return (-998,)  # pinned unsupported equality family is always awake
    trees = set()
    for target in (int(model.eq_obj1id[idx]), int(model.eq_obj2id[idx])):
      if target < 0:
        continue  # static equality operands do not wake the equality
      resolved = _object_trees(model, target_type, target, mujoco)
      if resolved is False:
        return False
      if resolved is None or resolved == (-998,):
        return (-998,)
      if resolved != (-999,):
        trees.update(resolved)
    if len(trees) > 2:
      return False
    return tuple(sorted(trees)) if trees else (-999,)
  elif typ == int(O.mjOBJ_SENSOR):
    # mj_sleepState(mjOBJ_SENSOR) recursively evaluates the referenced sensor.
    # The fixed policy stores this as a sensor dependency rather than guessing
    # from the referenced sensor's primary object alone.
    return (-997, idx) if 0 <= idx < int(model.nsensor) else False
  elif typ == int(O.mjOBJ_ACTUATOR):
    if not 0 <= idx < int(model.nu):
      return False
    T = mujoco.mjtTrn
    trn = int(model.actuator_trntype[idx])
    target = int(model.actuator_trnid[idx, 0])
    if trn in (int(T.mjTRN_JOINT), int(T.mjTRN_JOINTINPARENT)):
      return _object_trees(model, O.mjOBJ_JOINT, target, mujoco)
    if trn == int(T.mjTRN_TENDON):
      return _object_trees(model, O.mjOBJ_TENDON, target, mujoco)
    if trn == int(T.mjTRN_SITE):
      return _object_trees(model, O.mjOBJ_SITE, target, mujoco)
    if trn == int(T.mjTRN_BODY):
      return _object_trees(model, O.mjOBJ_BODY, target, mujoco)
    if trn == int(T.mjTRN_SLIDERCRANK):
      other = int(model.actuator_trnid[idx, 1])
      a = _object_trees(model, O.mjOBJ_SITE, target, mujoco)
      b = _object_trees(model, O.mjOBJ_SITE, other, mujoco)
      if a is False or b is False or a is None or b is None:
        return False
      return tuple(sorted(set(a) | set(b)))
    if trn == int(T.mjTRN_UNDEFINED):
      return (-998,)  # known object with the pinned always-awake state
    return False
  else:
    return False

  if body < 0 or body >= body_tree.size:
    return False
  tree = int(body_tree[body])
  return (tree,) if tree >= 0 else ()


def lower_sensor_sleep_policy(model):
  """Return immutable fixed-width sleep dependencies for each sensor.

  ``treeids`` has shape ``[nsensor, 2 objects, 2 trees]`` and ``counts`` has
  shape ``[nsensor, 2]``. Count -1 is UNKNOWN, -2 is known asleep, -3 is known
  awake, -4 references another sensor, 0 is STATIC, and positive counts name
  dynamic trees. Unsupported composite object families are marked in
  ``always_awake``.
  """
  import mujoco

  nsensor = int(model.nsensor)
  treeids = np.full((nsensor, 2, 2), -1, dtype=np.int32)
  counts = np.zeros((nsensor, 2), dtype=np.int32)
  always_awake = np.zeros((nsensor,), dtype=np.int32)
  sensor_refs = np.full((nsensor, 2), -1, dtype=np.int32)
  forced = {
      int(mujoco.mjtSensor.mjSENS_USER),
      int(mujoco.mjtSensor.mjSENS_PLUGIN),
      int(mujoco.mjtSensor.mjSENS_RANGEFINDER),
  }
  for sensor in range(nsensor):
    typ = int(model.sensor_type[sensor])
    objtype = int(model.sensor_objtype[sensor])
    reftype = int(model.sensor_reftype[sensor])
    if (typ in forced
        or (typ == int(mujoco.mjtSensor.mjSENS_CONTACT)
            and (objtype == int(mujoco.mjtObj.mjOBJ_SITE)
                 or reftype == int(mujoco.mjtObj.mjOBJ_SITE)))):
      always_awake[sensor] = 1
      continue
    for which, (kind, objid) in enumerate((
        (objtype, int(model.sensor_objid[sensor])),
        (reftype, int(model.sensor_refid[sensor])),
    )):
      trees = _object_trees(model, kind, objid, mujoco)
      if trees is False:
        always_awake[sensor] = 1
        break
      if trees is None:
        counts[sensor, which] = -1
        continue
      if trees == (-998,):
        counts[sensor, which] = -3
        continue
      if trees == (-999,):
        counts[sensor, which] = -2
        continue
      if len(trees) == 2 and trees[0] == -997:
        counts[sensor, which] = -4
        sensor_refs[sensor, which] = trees[1]
        continue
      if len(trees) > 2:
        always_awake[sensor] = 1
        break
      counts[sensor, which] = len(trees)
      if trees:
        treeids[sensor, which, :len(trees)] = trees
  return (treeids, counts, always_awake, sensor_refs)


def sensor_awake_mask_cpu(tree_awake, treeids, counts, always_awake,
                          sensor_refs=None):
  """CPU oracle of pinned `mj_sensorSleepState` output-retention policy."""
  awake = np.asarray(tree_awake) != 0
  if awake.ndim == 1:
    awake = awake[None, :]
  treeids = np.asarray(treeids, dtype=np.int32)
  counts = np.asarray(counts, dtype=np.int32)
  always_awake = np.asarray(always_awake, dtype=bool)
  if treeids.shape != (counts.shape[0], 2, 2) or counts.shape[1] != 2:
    raise ValueError("sensor tree metadata must have shapes [nsensor,2,2]/[nsensor,2]")
  if always_awake.shape != (counts.shape[0],):
    raise ValueError("always_awake must have shape [nsensor]")
  if sensor_refs is None:
    sensor_refs = np.full((counts.shape[0], 2), -1, dtype=np.int32)
  sensor_refs = np.asarray(sensor_refs, dtype=np.int32)
  if sensor_refs.shape != (counts.shape[0], 2):
    raise ValueError("sensor_refs must have shape [nsensor,2]")
  out = np.zeros((awake.shape[0], counts.shape[0]), dtype=np.int32)
  for world in range(awake.shape[0]):
    visiting = set()

    def object_state(sensor, which):
      count = int(counts[sensor, which])
      if count == -1:  # mjOBJ_UNKNOWN has mjS_AWAKE
        return True, False
      if count == -2:
        return False, True
      if count == -3:
        return True, False
      if count == -4:
        target = int(sensor_refs[sensor, which])
        return sensor_state(target), not sensor_state(target)
      if count == 0:  # static is neither awake nor asleep
        return False, False
      ids = treeids[sensor, which, :count]
      is_awake = bool(np.any(awake[world, ids]))
      return is_awake, not is_awake

    def sensor_state(sensor):
      if sensor < 0 or sensor >= counts.shape[0] or sensor in visiting:
        return True  # invalid/cyclic metadata fails open
      if always_awake[sensor]:
        return True
      visiting.add(sensor)
      obj_awake, obj_asleep = object_state(sensor, 0)
      ref_awake, ref_asleep = object_state(sensor, 1)
      visiting.remove(sensor)
      if counts[sensor, 0] == -1 and counts[sensor, 1] == -1:
        return True
      if counts[sensor, 0] == -1:
        return not ref_asleep
      if counts[sensor, 1] == -1:
        return not obj_asleep
      return obj_awake or ref_awake

    for sensor in range(counts.shape[0]):
      out[world, sensor] = int(sensor_state(sensor))
  return out


class SensorSleepPolicy:
  """Fixed-shape MPS kernel producing per-world sensor awake flags."""

  def __init__(self, model, batch_size, device="mps"):
    self._workspace_sizes = sensor_sleep_workspace_sizes(model, batch_size)
    batch_size = int(batch_size)
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("sensor sleep policy requires MPS compile_shader")
    self.batch_size = batch_size
    self.nsensor = int(model.nsensor)
    self.ntree = int(model.ntree)
    self.device = torch.device(device)
    if self.device.type != "mps":
      raise ValueError("SensorSleepPolicy requires an MPS device")
    treeids, counts, always, sensor_refs = lower_sensor_sleep_policy(model)
    expected = ((self.nsensor, 2, 2), (self.nsensor, 2),
                (self.nsensor,), (self.nsensor, 2))
    actual = tuple(np.asarray(value).shape for value in
                   (treeids, counts, always, sensor_refs))
    if actual != expected:
      raise ValueError(
          f"lowered sensor sleep arrays have shapes {actual}, expected {expected}")
    self.treeids = torch.as_tensor(treeids, dtype=torch.int32, device=self.device)
    self.counts = torch.as_tensor(counts, dtype=torch.int32, device=self.device)
    self.always = torch.as_tensor(always, dtype=torch.int32, device=self.device)
    self.sensor_refs = torch.as_tensor(
        sensor_refs, dtype=torch.int32, device=self.device)
    self.out = torch.empty((self.batch_size, self.nsensor), dtype=torch.int32,
                           device=self.device)
    self.dims = torch.tensor([self.batch_size, self.nsensor, max(self.ntree, 1)],
                             dtype=torch.int32, device=self.device)
    self._kernel = torch.mps.compile_shader(_SHADER.read_text()).sensor_awake_mask

  def run_device(self, tree_awake, *, out=None):
    import torch
    if (not isinstance(tree_awake, torch.Tensor)
        or tree_awake.device.type != "mps"
        or tree_awake.dtype != torch.int32
        or tuple(tree_awake.shape) != (self.batch_size, max(self.ntree, 1))
        or not tree_awake.is_contiguous()):
      raise ValueError("tree_awake must be contiguous MPS int32 [batch, ntree]")
    result = self.out if out is None else out
    if (not isinstance(result, torch.Tensor) or result.device.type != "mps"
        or result.dtype != torch.int32
        or tuple(result.shape) != (self.batch_size, self.nsensor)
        or not result.is_contiguous()):
      raise ValueError("out must be contiguous MPS int32 [batch, nsensor]")
    self._kernel(tree_awake.reshape(-1), self.treeids.reshape(-1),
                 self.counts.reshape(-1), self.always,
                 self.sensor_refs.reshape(-1), self.dims,
                 result.reshape(-1),
                 threads=(self.batch_size,),
                 group_size=(1,))
    return result
