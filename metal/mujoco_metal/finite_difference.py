# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native finite-difference derivatives built from device simulation queries.

The query transaction below snapshots unique Torch storage in place and the
mutable bookkeeping that owns it. It never reads numerical device values back
to the host. It is deliberately independent of the simulation's public
snapshot API, whose serialized checkpoint format is host-owned.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import fields, is_dataclass
import math
from numbers import Real
import inspect

import numpy as np

try:
  import torch
except ImportError:  # pragma: no cover - exercised by Torch-free imports
  torch = None


_I32_MAX = (1 << 31) - 1


def _storage_key(value):
  if torch is not None and isinstance(value, torch.Tensor):
    storage = value.untyped_storage()
    return (str(value.device), storage.data_ptr(), value.dtype), storage.nbytes()
  if isinstance(value, np.ndarray):
    root = value
    while isinstance(root.base, np.ndarray):
      root = root.base
    minimum, maximum = 0, root.dtype.itemsize
    if root.size:
      for size, stride in zip(root.shape, root.strides):
        extent = max(size - 1, 0) * stride
        minimum += min(extent, 0)
        maximum += max(extent, 0)
    return ("numpy", root.__array_interface__["data"][0]), maximum - minimum
  return None, 0


def _walk(value, seen, storages, owners):
  """Find mutable storage and owned program objects without following models."""
  ident = id(value)
  if ident in seen:
    return
  seen.add(ident)
  if torch is not None and isinstance(value, torch.Tensor):
    key, size = _storage_key(value)
    storages.setdefault(key, size)
    return
  if _device_plugin(value):
    return
  if isinstance(value, np.ndarray):
    if not value.flags.writeable:
      return
    key, size = _storage_key(value)
    storages.setdefault(key, size)
    return
  if isinstance(value, dict):
    for item in value.values():
      _walk(item, seen, storages, owners)
    return
  if isinstance(value, (list, tuple, set)):
    for item in value:
      _walk(item, seen, storages, owners)
    return
  if is_dataclass(value) and not isinstance(value, type):
    owners.append(value)
    for field in fields(value):
      _walk(getattr(value, field.name), seen, storages, owners)
    return
  module = type(value).__module__
  attrs = _instance_attrs(value)
  if module.startswith("mujoco_metal.") and attrs is not None:
    owners.append(value)
    for item in attrs.values():
      _walk(item, seen, storages, owners)


def _instance_attrs(value):
  """Read Python-owned attributes without invoking dynamic __getattr__ hooks."""
  if not type(value).__module__.startswith("mujoco_metal."):
    return None
  try:
    return object.__getattribute__(value, "__dict__")
  except AttributeError:
    return None


def estimate_transaction_bytes(sim):
  """Return bytes needed for the transaction's owned device checkpoint."""
  seen, storages, owners = set(), {}, []
  _walk(vars(sim), seen, storages, owners)
  return sum(storages.values()) + sum(
      _plugin_snapshot_bytes(plugin)
      for plugin in getattr(sim, "_native_plugins", ()))


def _device_plugin(value):
  # Metal shader libraries/kernels expose dynamic __getattr__ hooks that may
  # compile a function when probed. Plugin capability discovery must never
  # trigger those hooks while walking a preflight graph.
  snapshot = inspect.getattr_static(value, "device_snapshot", None)
  restore = inspect.getattr_static(value, "restore_device", None)
  return callable(snapshot) and callable(restore)


def _plugin_snapshot_bytes(plugin):
  """Require a preallocation bound for plugin-owned rollback payloads."""
  method = inspect.getattr_static(plugin, "device_snapshot_bytes", None)
  if not callable(method):
    raise TypeError(
        "native derivative plugins must declare device_snapshot_bytes()")
  if "device_snapshot_bytes" in vars(plugin):
    bound = method
  else:
    bound = method.__get__(plugin, type(plugin)) if hasattr(method, "__get__") else method
  size = bound()
  if isinstance(size, bool) or not isinstance(size, int) or size < 0:
    raise ValueError("plugin device_snapshot_bytes() must return nonnegative int")
  return size


class DeviceQueryTransaction(AbstractContextManager):
  """Reusable in-place rollback for a complete simulation finite-difference query.

  The first pass computes the complete clone requirement before allocating any
  snapshot storage. Plugin implementations must explicitly provide
  ``device_snapshot`` and ``restore_device``; host-only callbacks cannot
  participate in a native transaction.
  """

  def __init__(self, sim, extra_bytes=0):
    if torch is None:
      raise RuntimeError("PyTorch is required for native derivative queries")
    if isinstance(extra_bytes, bool) or not isinstance(extra_bytes, int) or extra_bytes < 0:
      raise ValueError("extra_bytes must be a nonnegative integer")
    self.sim = sim
    self.plugins = tuple(getattr(sim, "_native_plugins", ()))
    for plugin in self.plugins:
      if (not callable(getattr(plugin, "device_snapshot", None)) or
          not callable(getattr(plugin, "restore_device", None))):
        raise TypeError("native derivatives require device rollback methods on plugins")
    self.snapshot_bytes = estimate_transaction_bytes(sim)
    self.required_bytes = self.snapshot_bytes + extra_bytes
    limits = getattr(sim, "limits", None)
    budget = int(getattr(limits, "memory_budget_bytes", 1 << 62))
    if self.required_bytes > budget:
      raise ValueError("finite-difference memory budget exceeded before allocation")
    self._nodes = []
    self._memo = {}
    self._tensor_storage = {}
    self._restored_tensor_storages = set()
    self._numpy_storage = {}
    self._restored = {}
    stages = getattr(sim, "_forward_stages", None)
    self._stages = stages
    self._record = getattr(stages, "_record", None)
    self._coherent_inputs = []
    self._paired_qacc_version_state = []
    coupled = getattr(sim, "_last_coupled", None)
    coupled_qacc = (coupled.get("qacc") if isinstance(coupled, dict) else None)
    sensor_qacc = getattr(sim, "_last_sensor_qacc", None)
    for tensor, version_name in (
        (coupled_qacc, "_last_coupled_qacc_version"),
        (sensor_qacc, "_last_sensor_qacc_version"),
    ):
      saved_version = getattr(sim, version_name, None)
      coherent = (isinstance(tensor, torch.Tensor)
                  and isinstance(saved_version, int)
                  and int(tensor._version) == saved_version)
      self._paired_qacc_version_state.append(
          (tensor, version_name, saved_version, bool(coherent)))
    if self._record is not None:
      for name, value in self._record.input_tensors.items():
        expected = self._record.input_versions.get(name)
        if expected is not None and stages._version(value) == expected:
          self._coherent_inputs.append((name, value))
    self._capture(vars(sim))
    self._plugin_state = [(plugin, plugin.device_snapshot()) for plugin in self.plugins]
    self._entered = False

  def _capture(self, value):
    ident = id(value)
    if ident in self._memo:
      return self._memo[ident]
    if isinstance(value, torch.Tensor):
      key, storage_bytes = _storage_key(value)
      saved_storage = self._tensor_storage.get(key)
      if saved_storage is None:
        count = storage_bytes // value.element_size()
        flat = torch.as_strided(value, (count,), (1,), storage_offset=0)
        saved_storage = flat.detach().clone()
        self._tensor_storage[key] = saved_storage
      node = ["tensor", value, (key, saved_storage, storage_bytes)]
    elif isinstance(value, np.ndarray):
      if not value.flags.writeable:
        node = ["leaf", value, None]
      else:
        root = value
        while isinstance(root.base, np.ndarray):
          root = root.base
        key = ("numpy", root.__array_interface__["data"][0])
        snapshot = self._numpy_storage.get(key)
        if snapshot is None:
          minimum, maximum = 0, root.dtype.itemsize
          if root.size:
            for size, stride in zip(root.shape, root.strides):
              extent = max(size - 1, 0) * stride
              minimum += min(extent, 0)
              maximum += max(extent, 0)
          snapshot = np.empty(maximum - minimum, dtype=np.uint8)
          root_view = np.ndarray(root.shape, dtype=root.dtype, buffer=snapshot,
                                 offset=-minimum, strides=root.strides)
          np.copyto(root_view, root)
          self._numpy_storage[key] = snapshot
        else:
          minimum = 0
          if root.size:
            for size, stride in zip(root.shape, root.strides):
              extent = max(size - 1, 0) * stride
              minimum += min(extent, 0)
        root_address = root.__array_interface__["data"][0]
        address = value.__array_interface__["data"][0]
        view_meta = (int(address - root_address - minimum), value.shape,
                     value.strides, value.dtype)
        node = ["array", value, (snapshot, view_meta)]
    elif isinstance(value, dict):
      children = {}
      node = ["dict", value, children]
      self._memo[ident] = node
      children.update((key, self._capture(child)) for key, child in value.items())
    elif isinstance(value, list):
      children = []
      node = ["list", value, children]
      self._memo[ident] = node
      children.extend(self._capture(child) for child in value)
    elif isinstance(value, tuple):
      children = []
      node = ["tuple", value, children]
      self._memo[ident] = node
      children.extend(self._capture(child) for child in value)
    elif isinstance(value, set):
      children = []
      node = ["set", value, children]
      self._memo[ident] = node
      children.extend((member, self._capture(member)) for member in value)
    elif _device_plugin(value):
      node = ["leaf", value, None]
    elif is_dataclass(value) and not isinstance(value, type):
      children = {}
      node = ["record", value, children]
      self._memo[ident] = node
      children.update((f.name, self._capture(getattr(value, f.name)))
                      for f in fields(value))
    elif (type(value).__module__.startswith("mujoco_metal.")
          and _instance_attrs(value) is not None):
      node = ["object", value, None]
      self._memo[ident] = node
      node[2] = self._capture(_instance_attrs(value))
    else:
      node = ("leaf", value, None)
    self._memo[ident] = node
    self._nodes.append(node)
    return node

  def _restore(self, node):
    node_id = id(node)
    if node_id in self._restored:
      return self._restored[node_id]
    kind, original, saved = node
    if kind == "tensor":
      self._restored[node_id] = original
      key, saved_storage, storage_bytes = saved
      if key not in self._restored_tensor_storages:
        count = storage_bytes // original.element_size()
        destination = torch.as_strided(original, (count,), (1,), storage_offset=0)
        destination.copy_(saved_storage)
        self._restored_tensor_storages.add(key)
    elif kind == "array":
      self._restored[node_id] = original
      snapshot, (offset, shape, strides, dtype) = saved
      view = np.ndarray(shape, dtype=dtype, buffer=snapshot,
                        offset=offset, strides=strides)
      np.copyto(original, view)
    elif kind == "dict":
      self._restored[node_id] = original
      original.clear()
      original.update((key, self._restore(child)) for key, child in saved.items())
    elif kind == "list":
      self._restored[node_id] = original
      original[:] = [self._restore(child) for child in saved]
    elif kind == "tuple":
      self._restored[node_id] = original
      for child in saved:
        self._restore(child)
    elif kind == "set":
      self._restored[node_id] = original
      members = [self._restore(child) for _, child in saved]
      original.clear()
      original.update(members)
    elif kind == "record":
      self._restored[node_id] = original
      for name, child in saved.items():
        object.__setattr__(original, name, self._restore(child))
    elif kind == "object":
      self._restored[node_id] = original
      attrs = dict(self._restore(saved))
      vars(original).clear()
      vars(original).update(attrs)
    else:
      self._restored[node_id] = original
    return original

  def restore(self):
    """Restore the entry state while retaining every original object/storage."""
    self._restored.clear()
    self._restored_tensor_storages.clear()
    self._restore(self._memo[id(vars(self.sim))])
    # In-place tensor restoration advances PyTorch mutation counters. Refresh
    # only source tags that were coherent at transaction entry; preserve stale
    # tags and any replaced tensor identities exactly as captured.
    for tensor, version_name, saved_version, coherent in (
        self._paired_qacc_version_state):
      if not coherent:
        continue
      if version_name == "_last_coupled_qacc_version":
        coupled = getattr(self.sim, "_last_coupled", None)
        current = coupled.get("qacc") if isinstance(coupled, dict) else None
      else:
        current = getattr(self.sim, "_last_sensor_qacc", None)
      if current is tensor:
        setattr(self.sim, version_name, int(tensor._version))
      else:
        setattr(self.sim, version_name, saved_version)
    if (self._record is not None and self._stages is not None and
        getattr(self._stages, "_record", None) is self._record):
      for name, value in self._coherent_inputs:
        if self._record.input_tensors.get(name) is value:
          self._stages.capture_input(self._record, name, value)
    errors = []
    for plugin, saved in reversed(self._plugin_state):
      try:
        plugin.restore_device(saved)
      except Exception as error:  # restore all plugins before surfacing failure
        errors.append(error)
    if errors:
      raise RuntimeError("native derivative plugin rollback failed") from errors[0]

  def __enter__(self):
    self._entered = True
    return self

  def __exit__(self, exc_type, exc, traceback):
    self.restore()
    return False


def _eps(value):
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
    raise ValueError("eps must be a finite nonzero float32 scalar")
  with np.errstate(over="ignore", under="ignore", invalid="ignore"):
    result = np.float32(value)
  if not math.isfinite(float(result)) or result <= 0:
    raise ValueError("eps must be a positive finite float32 scalar")
  return float(result)


def _dimensions(sim):
  model = sim._mjmodel
  return (int(model.nq), int(model.nv), int(model.na), int(model.nu),
          int(model.nsensordata))


def _check_dims(batch, *dimensions):
  if any(x < 0 or x > _I32_MAX for x in (batch, *dimensions)):
    raise OverflowError("finite-difference indexing exceeds signed int32")


def _check_product(*dimensions):
  product = 1
  for dimension in dimensions:
    product *= int(dimension)
    if product > _I32_MAX:
      raise OverflowError("finite-difference tensor indexing exceeds signed int32")
  return product


def _preflight(sim, bytes_required):
  limits = getattr(sim, "limits", None)
  budget = int(getattr(limits, "memory_budget_bytes", 1 << 62))
  if bytes_required > budget:
    raise ValueError("finite-difference memory budget exceeded before allocation")


def _position_program_bytes(sim):
  program = getattr(sim, "_position_queries", None)
  if program is not None:
    return int(program.required_bytes)
  model = sim._mjmodel
  nj = int(model.njnt)
  b = int(sim.batch_size)
  nq, nv = int(model.nq), int(model.nv)
  return (max(3 * nj, 1) + 4 + 1 + 3 + 1 +
          max(b * max(nq, nv), 1) + b) * 4


def _zero_failed_output_rows(outputs):
  """Turn numerical differencing overflow into a failed, zeroed world row."""
  status = outputs["status"]
  invalid = torch.zeros_like(status, dtype=torch.bool)
  for name, value in outputs.items():
    if name != "status" and value.numel():
      invalid |= ~torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
  status.copy_(torch.maximum(status, torch.where(
      invalid, torch.full_like(status, 2), torch.zeros_like(status))))
  failed = status != 0
  for name, value in outputs.items():
    if name != "status" and value.numel():
      value.masked_fill_(failed.reshape((failed.shape[0],) + (1,) * (value.ndim - 1)), 0)


def _query_step(sim):
  status = sim.step()
  sensors = getattr(sim, "_sensors", None)
  sensor_low = getattr(sensors, "_s_state_out_low", None)
  if sensor_low is None:
    sensor_low = (sim._state._qvel[:, :0] if getattr(sim, "_sensordata", None) is None
                  else torch.zeros_like(sim._sensordata))
  return status.clone(), sim._state._qpos.clone(), sim._state._qvel.clone(), (
      sim._state._act.clone() if sim._state._act is not None else
      sim._state._qvel[:, :0]), (
      sim._sensordata.clone() if getattr(sim, "_sensordata", None) is not None else
      sim._state._qvel[:, :0]), sensor_low.clone()


def _invalidate_for_query(sim):
  sim._state._generation += 1
  sim._invalidate_forward_stage_record()
  sim._assembled_system_valid = False


def _state_diff(sim, first, second, h, *, centered=False):
  from mujoco_metal.native_api import mj_differentiatePos
  result = mj_differentiatePos(sim, first[0], second[0], h)
  return (torch.cat((result["qvel"], (second[1] - first[1]) / h,
                     (second[2] - first[2]) / h), dim=-1),
          result["status"])


def _column(sim, transaction, field, index, delta, *, skip, sensor=True):
  transaction.restore()
  state = sim._state
  input_status = None
  if field == "qpos":
    from mujoco_metal.native_api import mj_integratePos
    tangent = torch.zeros((sim.batch_size, int(sim._mjmodel.nv)),
                         dtype=torch.float32, device=state.device)
    tangent[:, index] = float(delta)
    integrated = mj_integratePos(sim, state._qpos, tangent, 1.0)
    state._qpos.copy_(integrated["qpos"])
    input_status = integrated["status"]
  elif field == "ctrl":
    sim._control[:, index].add_(float(delta))
  else:
    attr = {"qvel": "_qvel", "act": "_act", "qacc": "_qacc"}[field]
    getattr(state, attr)[:, index].add_(float(delta))
  _invalidate_for_query(sim)
  if field == "qacc":
    details = sim.inverse_skip(skipsensor=False)
    force = details["qfrc_inverse"]
    sensors = getattr(sim, "_sensors", None)
    sensor_low = getattr(sensors, "_s_state_out_low", None)
    if sensor_low is None:
      sensor_low = (sim._state._qvel[:, :0] if sim._sensordata is None
                    else torch.zeros_like(sim._sensordata))
    return (details["status"].clone(), force,
            sim._state._qpos.clone(), sim._state._qvel.clone(),
            sim._state._act.clone() if sim._state._act is not None else
            sim._state._qvel[:, :0],
            sim._sensordata.clone() if sim._sensordata is not None else
            sim._state._qvel[:, :0], sensor_low.clone())
  result = _query_step(sim)
  if input_status is not None:
    result = (torch.maximum(result[0], input_status), *result[1:])
  return result


def _inverse_force_delta(base_components, candidate_components, *,
                         base_actuation=None, candidate_actuation=None):
  """Difference inverse force constituents before combining common values."""
  delta = (candidate_components["mass_qacc"] -
           base_components["mass_qacc"])
  delta.add_(candidate_components["bias"] - base_components["bias"])
  delta.add_(candidate_components["bias_low"] - base_components["bias_low"])
  delta.sub_(candidate_components["passive"] - base_components["passive"])
  delta.sub_(candidate_components["constraint"] - base_components["constraint"])
  base_constraint_low = base_components.get("constraint_low")
  candidate_constraint_low = candidate_components.get("constraint_low")
  if base_constraint_low is not None or candidate_constraint_low is not None:
    if base_constraint_low is None or candidate_constraint_low is None:
      raise ValueError("inverse constraint low words must be present for both samples")
    delta.sub_(candidate_constraint_low - base_constraint_low)
  if base_actuation is not None:
    if candidate_actuation is None:
      raise ValueError("candidate actuation is required with a base actuation")
    delta.sub_(candidate_actuation - base_actuation)
  return delta


def _difference_candidates(sim, baseline, plus, minus, eps, centered,
                           can_plus=None, can_minus=None):
  """Choose centered/one-sided device differences independently per world."""
  b, nv, na = sim.batch_size, int(sim._mjmodel.nv), int(sim._mjmodel.na)
  needs_back = bool(centered) or can_minus is not None
  can_plus = (torch.ones((b,), dtype=torch.bool, device=sim.device)
              if can_plus is None else can_plus)
  can_minus = (torch.full((b,), bool(centered), dtype=torch.bool,
                          device=sim.device)
               if can_minus is None else can_minus)
  both = can_plus & can_minus
  if centered:
    ycenter, status_center = _state_diff(sim, minus[1:4], plus[1:4], 2 * eps)
    scenter = ((plus[4] - minus[4]) + (plus[5] - minus[5])) / (2 * eps)
  else:
    status_center = None
  yfwd, status_fwd = _state_diff(sim, baseline[1:4], plus[1:4], eps)
  sfwd = ((plus[4] - baseline[4]) +
          (plus[5] - baseline[5])) / eps
  if not centered:
    ycenter, scenter = torch.zeros_like(yfwd), torch.zeros_like(sfwd)
  if needs_back:
    yback, status_back = _state_diff(sim, minus[1:4], baseline[1:4], eps)
    sback = ((baseline[4] - minus[4]) +
             (baseline[5] - minus[5])) / eps
  else:
    yback, sback = torch.zeros_like(yfwd), torch.zeros_like(sfwd)
    status_back = torch.zeros_like(baseline[0])
  if centered:
    use_center = both
    use_fwd = can_plus & ~can_minus
    use_back = can_minus & ~can_plus
  else:
    use_center = torch.zeros_like(can_plus)
    use_fwd = can_plus
    use_back = ~can_plus & can_minus
  y = torch.where(use_center[:, None], ycenter,
      torch.where(use_fwd[:, None], yfwd,
        torch.where(use_back[:, None], yback, torch.zeros_like(yfwd))))
  s = torch.where(use_center[:, None], scenter,
      torch.where(use_fwd[:, None], sfwd,
        torch.where(use_back[:, None], sback, torch.zeros_like(sfwd))))
  position_status = torch.where(use_center, status_center if centered else baseline[0],
      torch.where(use_fwd, status_fwd,
        torch.where(use_back, status_back, torch.zeros_like(baseline[0]))))
  valid = baseline[0] == 0
  valid &= (~can_plus) | (plus[0] == 0)
  valid &= (~can_minus) | (minus[0] == 0)
  valid &= position_status == 0
  return (torch.where(valid[:, None], y, torch.zeros_like(y)),
          torch.where(valid[:, None], s, torch.zeros_like(s)),
          torch.maximum(baseline[0], torch.maximum(
              torch.where(can_plus, plus[0], torch.zeros_like(plus[0])),
              torch.maximum(
                  torch.where(can_minus, minus[0], torch.zeros_like(minus[0])),
                  position_status))))


def mjd_stepFD(sim, eps=1e-6, flg_centered=False):
  """Differentiate one native step; arrays use source transposed input rows.

  Returns owned ``DyDq/DyDv/DyDa/DyDu``, ``DsDq/DsDv/DsDa/DsDu`` and
  int32 ``status[B]``. Epsilon is rounded to float32 before differencing.
  """
  e = _eps(eps)
  if not isinstance(flg_centered, (bool, np.bool_)):
    raise TypeError("flg_centered must be boolean")
  nq, nv, na, nu, ns = _dimensions(sim)
  b, ndx = int(sim.batch_size), 2 * nv + na
  _check_dims(b, nq, nv, na, nu, ns, ndx, b * max(ndx, 1), b * max(ns, 1))
  for output_width, input_width in ((nv, ndx), (na, ndx), (nu, ndx),
                                   (nv, ns), (na, ns), (nu, ns)):
    _check_product(b, output_width, input_width)
  _check_product(b, 2 * nq + 8 * nv + 4 * na + nu + 4 * ns)
  if int(sim._mjmodel.nhistory):
    raise ValueError("pinned mjd_stepFD does not support actuator delays")
  est = (b * (2 * nv + na + nu) * ndx +
         b * (2 * nv + na + nu) * ns + b) * 4
  # One baseline/candidate state and sensors plus the complete reusable
  # simulation checkpoint are charged before the first result/snapshot alloc.
  est += b * (2 * nq + 8 * nv + 4 * na + nu + 4 * ns) * 4
  # Candidate sensor residuals are retained separately until the high-word
  # differences have been formed, avoiding rounding them back into the large
  # common float32 sample first.
  est += b * 3 * ns * 4
  _preflight(sim, estimate_transaction_bytes(sim) + est +
             _position_program_bytes(sim))
  # Position perturbations and state differences share this reusable native
  # topology/kernel. Capture it in the transaction before the first query so
  # restores do not discard it and trigger compilation for every column.
  from mujoco_metal.native_api import _position_query_program
  _position_query_program(sim)
  with DeviceQueryTransaction(sim, est) as tx:
    out = {name: torch.zeros((b, width, ndx), dtype=torch.float32,
           device=sim.device) for name, width in (
               ("DyDq", nv), ("DyDv", nv), ("DyDa", na), ("DyDu", nu))}
    out.update({name: torch.zeros((b, width, ns), dtype=torch.float32,
                 device=sim.device) for name, width in (
                     ("DsDq", nv), ("DsDv", nv), ("DsDa", na), ("DsDu", nu))})
    out["status"] = torch.zeros((b,), dtype=torch.int32, device=sim.device)
    input_control = sim._control.clone()
    baseline = _query_step(sim)
    out["status"].copy_(baseline[0])
    for field, width, names in (("qpos", nv, ("DyDq", "DsDq")),
                                ("qvel", nv, ("DyDv", "DsDv")),
                                ("act", na, ("DyDa", "DsDa")),
                                ("ctrl", nu, ("DyDu", "DsDu"))):
      if width == 0:
        continue
      for i in range(width):
        tx.restore()
        control = field == "ctrl"
        can_plus = can_minus = None
        if control and bool(sim._mjmodel.actuator_ctrllimited[i]):
          lo, hi = map(float, sim._mjmodel.actuator_ctrlrange[i])
          values = input_control[:, i]
          can_plus = ((values >= lo) & (values <= hi) &
                      (values + e >= lo) & (values + e <= hi))
          interval_minus = ((values >= lo) & (values <= hi) &
                            (values - e >= lo) & (values - e <= hi))
          can_minus = (interval_minus & (bool(flg_centered) | ~can_plus))
        plus = _column(sim, tx, field, i, e, skip=None)
        minus = _column(sim, tx, field, i, -e, skip=None)
        ydiff, sdiff, column_status = _difference_candidates(
            sim, baseline, plus, minus, e, bool(flg_centered),
            can_plus=can_plus, can_minus=can_minus)
        out["status"] = torch.maximum(out["status"], column_status)
        out[names[0]][:, i, :] = ydiff
        out[names[1]][:, i, :] = sdiff
  _zero_failed_output_rows(out)
  return out


def mjd_transitionFD(sim, eps=1e-6, flg_centered=False):
  """Return control-theory ``A/B/C/D`` from native step finite differences."""
  model = sim._mjmodel
  import mujoco
  if int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
    raise ValueError("pinned mjd_transitionFD does not support RK4")
  if int(model.nhistory):
    raise ValueError("pinned mjd_transitionFD does not support actuator delays")
  nq, nv, na, nu, ns = _dimensions(sim)
  nx = 2 * nv + na
  b = int(sim.batch_size)
  _check_dims(b, nx, nu, ns, b * max(nx, 1), b * max(ns, 1))
  for rows, cols in ((nx, nx), (nx, nu), (ns, nx), (ns, nu)):
    _check_product(b, rows, cols)
  row_bytes = (b * ((2 * nv + na + nu) * nx +
                    (2 * nv + na + nu) * ns + 1)) * 4
  matrix_bytes = (b * (nx * nx + nx * nu + ns * nx + ns * nu + 1)) * 4
  scratch_bytes = b * (2 * nq + 8 * nv + 4 * na + nu + 7 * ns) * 4
  _preflight(sim, estimate_transaction_bytes(sim) + row_bytes +
             matrix_bytes + scratch_bytes + _position_program_bytes(sim))
  rows = mjd_stepFD(sim, eps=eps, flg_centered=flg_centered)
  A = torch.cat((rows["DyDq"], rows["DyDv"], rows["DyDa"]), dim=1).transpose(1, 2).contiguous()
  B = rows["DyDu"].transpose(1, 2).contiguous()
  C = torch.cat((rows["DsDq"], rows["DsDv"], rows["DsDa"]), dim=1).transpose(1, 2).contiguous()
  D = rows["DsDu"].transpose(1, 2).contiguous()
  result = {"A": A, "B": B, "C": C, "D": D, "status": rows["status"]}
  _zero_failed_output_rows(result)
  return result


def mjd_inverseFD(sim, eps=1e-6, flg_actuation=False):
  """Differentiate native inverse dynamics and optional packed mass entries."""
  e = _eps(eps)
  if not isinstance(flg_actuation, (bool, np.bool_)):
    raise TypeError("flg_actuation must be boolean")
  model = sim._mjmodel
  import mujoco
  if int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
    raise ValueError("pinned mjd_inverseFD does not support RK4")
  if int(model.opt.noslip_iterations):
    raise ValueError("pinned mjd_inverseFD does not support the noslip solver")
  nq, nv, na, nu, ns = _dimensions(sim)
  b = int(sim.batch_size)
  nM = int(model.nM)
  _check_dims(b, nq, nv, ns, nM, b * max(nv, 1) ** 2)
  _check_product(b, max(nv, 1), max(nv, nM, ns, 1))
  _check_product(b, nv, nv)
  from mujoco_metal.native_api import mj_fullM
  from mujoco_metal.forward_stages import ForwardStage
  extra = (b * (3 * nv * nv + 3 * nv * ns + nv * nM) + b) * 4
  # Baseline and candidate each retain mass, bias, passive and constraint
  # vectors so force differences can be formed before their large common
  # values are combined.
  extra += b * (2 * nv * nv + 14 * nv + ns) * 4
  extra += _inverse_constraint_pair_workspace_bytes(sim)
  extra += b * 2 * ns * 4  # baseline and current paired sensor low words
  extra += _inverse_discrete_pair_workspace_bytes(sim)
  sensor_enabled = _inverse_fd_sensor_enabled(sim)
  # A pre-step query can lazily materialize both the public sensor output and
  # the raw history-recording backing while inverseSkip evaluates its stages.
  # They are not present in estimate_transaction_bytes when their owner fields
  # are None at entry, so charge those exact planes before entering the query.
  if ns:
    has_sensor_program = (sensor_enabled and (
        getattr(sim, "_sensors", None) is not None or
        getattr(sim, "_raw_sensor_program", None) is not None))
    extra += b * ns * 4 * (
        int(getattr(sim, "_sensordata", None) is None) +
        int(has_sensor_program and getattr(sim, "_raw_sensordata", None) is None))
    if has_sensor_program:
      # _store_step_sensors peaks at previous, previous-raw, stage output,
      # raw-retained and history-selected planes while preserving skipped
      # stages. Sensor program workspaces themselves are already persistent
      # and included in the transaction graph.
      extra += b * ns * 5 * 4
      if getattr(sim, "_has_user_plugin_sensors", False):
        sensor_plugin_bytes = [
            _plugin_snapshot_bytes(plugin)
            for plugin in getattr(sim, "_native_plugins", ())
            if getattr(getattr(plugin, "plugin_type", None), "value", None)
            == "sensor"]
        # _store_step_sensors snapshots the complete sensor-plugin set and
        # then one plugin again around its isolated callback.
        extra += sum(sensor_plugin_bytes) + max(sensor_plugin_bytes, default=0)
  has_sensor_qacc_low = (sensor_enabled and
                         _inverse_fd_query_qacc_low(sim, clone=False) is not None)
  if has_sensor_qacc_low:
    extra += b * nv * 4
  if ns:
    # The pinned skip-stage loop carries the last sample's unrefreshed sensor
    # stages into the next column. Keep one independent high-word cursor in
    # addition to the immutable center sample and live sensor output.
    extra += b * ns * 4
    # Its low word has the same three live roles: center, preceding sample,
    # and the currently captured sample.
    extra += b * ns * 4
  # With actuation subtraction enabled, mjd_inverseFD runs ACT after each
  # inverse sample. The next sample's ACC sensors read those just-produced
  # actuator force inputs, so retain their two simulation-owned planes across
  # per-column rollback. The outer transaction still restores caller state.
  if (flg_actuation and sensor_enabled and
      bool(getattr(sim, "_has_acc_sensors", False))):
    extra += b * (max(int(model.nu), 1) + max(nv, 1)) * 4
  _preflight(sim, estimate_transaction_bytes(sim) + extra +
             _position_program_bytes(sim))
  from mujoco_metal.native_api import _position_query_program
  _position_query_program(sim)
  with DeviceQueryTransaction(sim, extra) as tx:
    outputs = {name: torch.zeros((b, nv, nv), dtype=torch.float32, device=sim.device)
               for name in ("DfDq", "DfDv", "DfDa")}
    outputs.update({name: torch.zeros((b, nv, ns), dtype=torch.float32, device=sim.device)
                    for name in ("DsDq", "DsDv", "DsDa")})
    outputs["DmDq"] = torch.zeros((b, nv, nM), dtype=torch.float32, device=sim.device)
    outputs["status"] = torch.zeros((b,), dtype=torch.int32, device=sim.device)
    tx.restore()
    # MuJoCo's mjData always owns sensordata, even before the first forward
    # call.  Native Simulation allocates this backing lazily; materialize the
    # same zero-initialized state before inverseSkip so its stage-specific
    # sensor writes (and history reads) happen in the pinned order.
    _inverse_fd_prepare_sensor_storage(sim, ns)
    sensor_qacc_low = (_inverse_fd_query_qacc_low(sim)
                       if sensor_enabled else None)
    base_detail = sim.inverse_skip(skipsensor=not sensor_enabled,
                                   return_components=True,
                                   sensor_qacc_low=sensor_qacc_low)
    outputs["status"].copy_(base_detail["status"])
    base_components = base_detail["qfrc_inverse_components"]
    base_sensor = (_inverse_fd_sensor_sample(sim, base_detail, ns) if ns else
                   torch.empty((b, 0), dtype=torch.float32, device=sim.device))
    sensor_cursor = base_sensor.clone()
    base_sensor_low = _inverse_fd_sensor_low(sim, base_sensor)
    sensor_low_cursor = base_sensor_low.clone()
    dynamics = base_detail["record"].values[ForwardStage.VEL]["dynamics"]
    base_mass = mj_fullM(sim, dynamics=dynamics)
    base_actuator = None
    actuator_sensor_inputs = None
    if flg_actuation:
      base_actuator = sim.prepare_forward_actuation(
          base_detail["record"])["qfrc_actuator"].clone()
      actuator_sensor_inputs = _capture_inverse_fd_actuator_sensor_inputs(sim)
    dof_madr = np.asarray(model.dof_Madr, dtype=np.int64)
    dof_parent = np.asarray(model.dof_parentid, dtype=np.int64)
    packed = []
    for dof in range(nv):
      row = [dof]
      parent = int(dof_parent[dof])
      while parent >= 0:
        row.append(parent)
        parent = int(dof_parent[parent])
      expected = (int(dof_madr[dof + 1]) if dof + 1 < nv else nM) - int(dof_madr[dof])
      if len(row) != expected:
        raise ValueError("compiled qM ancestor map is inconsistent")
      # qM is ordered row by row: diagonal, then successive ancestors.
      packed.extend((dof, column) for column in row)
    # Match engine_derivative_fd.c exactly: acceleration, velocity, then
    # position. Although coordinates are restored after each sample, mjData's
    # stage outputs are not; in particular skipped sensor stages retain the
    # immediately preceding sample, not always the center sample.
    for field, name in (("qacc", "a"), ("qvel", "v"), ("qpos", "q")):
      for i in range(nv):
        tx.restore()
        _restore_inverse_fd_actuator_sensor_inputs(
            sim, actuator_sensor_inputs)
        # inverseSkip's POS/VEL skip flags mean the previously computed
        # stages remain in sensordata.  Re-seed that stage state after the
        # query rollback before asking it to update only the affected stages.
        _inverse_fd_prepare_sensor_storage(sim, ns, sensor_cursor)
        _inverse_fd_seed_sensor_low(sim, sensor_low_cursor)
        if field == "qpos":
          from mujoco_metal.native_api import mj_integratePos
          tangent = torch.zeros((b, nv), dtype=torch.float32, device=sim.device)
          tangent[:, i] = e
          integrated = mj_integratePos(sim, sim._state._qpos, tangent, 1.0)
          sim._state._qpos.copy_(integrated["qpos"])
          position_status = integrated["status"]
        else:
          position_status = torch.zeros((b,), dtype=torch.int32, device=sim.device)
          getattr(sim._state, {"qvel": "_qvel", "qacc": "_qacc"}[field])[:, i].add_(e)
        _invalidate_for_query(sim)
        skipstage = {
            "qpos": mujoco.mjtStage.mjSTAGE_NONE,
            "qvel": mujoco.mjtStage.mjSTAGE_POS,
            "qacc": mujoco.mjtStage.mjSTAGE_VEL,
        }[field]
        prefix_record = None
        if skipstage != mujoco.mjtStage.mjSTAGE_NONE:
          # Query rollback restores the entry-stage cache. Rebuild exactly the
          # prefix that pinned inverseSkip would retain in mjData, then ask it
          # to update only the stages affected by this perturbation.
          prepare_pos = getattr(sim, "prepare_forward_position", None)
          if not callable(prepare_pos):
            raise RuntimeError("inverse FD requires prepared POS stage support")
          # These calls rebuild only the cached dynamics prefix. Sensor
          # stages are evaluated by inverse_skip according to its selected
          # skipstage, just as in MuJoCo's mjd_inverseFD loop.
          prefix_record = prepare_pos(skipsensor=True)
          if skipstage == mujoco.mjtStage.mjSTAGE_VEL:
            prepare_vel = getattr(sim, "prepare_forward_velocity", None)
            if not callable(prepare_vel):
              raise RuntimeError("inverse FD requires prepared VEL stage support")
            # Simulation publishes VEL into the live coordinator record but
            # returns the stage-values dictionary. Keep the POS record for
            # inverse_skip's ownership/epoch validation; replacing it with
            # that dictionary bypasses the prepared-stage contract.
            prepare_vel(prefix_record, skipsensor=True)
        detail = sim.inverse_skip(skipstage=skipstage,
                                  skipsensor=not sensor_enabled,
                                  return_components=True,
                                  sensor_qacc_low=sensor_qacc_low,
                                  record=prefix_record)
        components = detail["qfrc_inverse_components"]
        sens = (_inverse_fd_sensor_sample(sim, detail, ns, copy=False) if ns else
                base_sensor)
        sens_low = _inverse_fd_sensor_low(sim, sens)
        sensor_cursor.copy_(sens)
        sensor_low_cursor = sens_low
        status = detail["status"]
        dynamics = detail["record"].values[ForwardStage.VEL]["dynamics"]
        actuator = None
        if flg_actuation:
          actuator = sim.prepare_forward_actuation(
              detail["record"])["qfrc_actuator"]
          actuator_sensor_inputs = _capture_inverse_fd_actuator_sensor_inputs(
              sim, into=actuator_sensor_inputs)
        force_delta = _inverse_force_delta(
            base_components, components, base_actuation=base_actuator,
            candidate_actuation=actuator)
        valid = ((base_detail["status"] == 0) & (status == 0) &
                 (position_status == 0))
        outputs["status"] = torch.maximum(outputs["status"],
            torch.maximum(status, position_status))
        outputs[f"DfD{name}"][:, i, :] = torch.where(
            valid[:, None], force_delta / e, torch.zeros_like(force_delta))
        if ns:
          outputs[f"DsD{name}"][:, i, :] = torch.where(
              valid[:, None],
              ((sens - base_sensor) + (sens_low - base_sensor_low)) / e,
              torch.zeros_like(sens))
        if field == "qpos":
          mass = mj_fullM(sim, dynamics=dynamics)
          for k, (r, c) in enumerate(packed):
            outputs["DmDq"][:, i, k] = torch.where(
                valid, (mass[:, r, c] - base_mass[:, r, c]) / e,
                torch.zeros_like(valid, dtype=torch.float32))
  _zero_failed_output_rows(outputs)
  return outputs


def _capture_inverse_fd_actuator_sensor_inputs(sim, *, into=None):
  """Retain ACT outputs consumed by the next pinned inverse ACC sample.

  MuJoCo's ``inverseSkip`` writes sensors before its optional
  ``mj_fwdActuation`` call. That ACT call leaves actuator force inputs live in
  ``mjData`` for the next perturbation. In Simulation the corresponding
  inputs to ACC sensor evaluation are these two reduced force planes.
  """
  if into is None:
    into = {}
  for name in ("_sen_act_force", "_sen_qfrc_act"):
    current = getattr(sim, name, None)
    if current is None:
      continue
    if (torch is None or not isinstance(current, torch.Tensor) or
        current.dtype != torch.float32 or not current.is_contiguous()):
      raise ValueError(f"inverse FD {name} must be contiguous float32")
    saved = into.get(name)
    if saved is None:
      into[name] = current.clone()
    elif (tuple(saved.shape) != tuple(current.shape) or
          saved.dtype != current.dtype or saved.device != current.device):
      raise ValueError(f"inverse FD {name} changed layout during query")
    else:
      saved.copy_(current)
  return into


def _restore_inverse_fd_actuator_sensor_inputs(sim, saved):
  """Seed the next sample with the preceding sample's completed ACT outputs."""
  if not saved:
    return
  for name, value in saved.items():
    current = getattr(sim, name, None)
    if current is None:
      # Lazy sensor scratch is allocated by inverseSkip/prepare ACT. Reusing
      # the owned cursor tensor avoids allocating a new plane each column.
      setattr(sim, name, value)
    elif (not isinstance(current, torch.Tensor) or
          tuple(current.shape) != tuple(value.shape) or
          current.dtype != value.dtype or current.device != value.device):
      raise ValueError(f"inverse FD {name} changed layout during query")
    elif current is not value:
      current.copy_(value)


def _inverse_fd_sensor_low(sim, sensordata):
  """Clone the low plane from the program that actually produced the sample.

  Inverse queries evaluate stored sensors through ``_raw_sensor_program``;
  when it is absent, the ordinary sensor program owns the sample. A program
  may exist before its optional low plane has been allocated.
  """
  if sensordata.numel() == 0:
    return torch.zeros_like(sensordata)
  if not _inverse_fd_sensor_enabled(sim):
    return torch.zeros_like(sensordata)
  program = _inverse_fd_sensor_program(sim)
  low = (getattr(program, "_s_state_out_low", None)
         if program is not None else None)
  if low is None:
    return torch.zeros_like(sensordata)
  if (not isinstance(low, torch.Tensor)
      or low.shape != sensordata.shape
      or low.dtype != sensordata.dtype
      or low.device != sensordata.device):
    raise ValueError("inverse FD sensor low plane does not match sensordata")
  result = low.clone()
  # History-backed values are selected from the retained ring after raw
  # sensor evaluation.  Their low word is not the raw sensor's current low
  # word: delayed sensors always read history, while interval sensors do so
  # on non-tick worlds.  Keep the pair aligned with the returned high sample.
  history = getattr(sim, "_history_program", None)
  entries = getattr(history, "entries", None)
  if history is not None and entries is not None and len(entries):
    model = sim._mjmodel
    delays = np.asarray(getattr(model, "sensor_delay", ()), dtype=np.float64)
    intervals = np.asarray(getattr(model, "sensor_interval", ()), dtype=np.float64)
    sensor_adr = np.asarray(getattr(model, "sensor_adr", ()), dtype=np.int64)
    sensor_dim = np.asarray(getattr(model, "sensor_dim", ()), dtype=np.int64)
    tick_mask = None
    for entry in np.asarray(entries, dtype=np.int32):
      kind, _, _, _, _, outadr = map(int, entry)
      if kind != 1:
        continue
      sensor = int(np.searchsorted(sensor_adr, outadr))
      if sensor >= sensor_adr.size or int(sensor_adr[sensor]) != outadr:
        continue
      width = int(sensor_dim[sensor])
      delay = float(delays[sensor]) if sensor < delays.size else 0.0
      interval = (float(intervals[sensor, 0])
                  if intervals.ndim == 2 and sensor < intervals.shape[0]
                  else 0.0)
      target = result[:, outadr:outadr + width]
      if delay > 0:
        target.zero_()
      elif interval > 0:
        if tick_mask is None:
          tick_mask = history.sensor_compute_mask(
              sim._state._history, sim._state._time)
        target.masked_fill_(~tick_mask[:, outadr:outadr + width], 0.0)
  return result


def _inverse_fd_sensor_program(sim):
  program = getattr(sim, "_raw_sensor_program", None)
  if program is None:
    program = getattr(sim, "_sensors", None)
  return program


def _inverse_fd_seed_sensor_low(sim, seed):
  """Restore the retained low plane before a skip-stage sensor update."""
  if seed is None:
    return
  program = _inverse_fd_sensor_program(sim)
  if program is None:
    return
  current = getattr(program, "_s_state_out_low", None)
  if current is None:
    # The low plane may be allocated lazily by the current sensor producer.
    # Reattach the owned cursor until that producer replaces or writes it.
    setattr(program, "_s_state_out_low", seed)
    return
  if (not isinstance(current, torch.Tensor)
      or tuple(current.shape) != tuple(seed.shape)
      or current.dtype != seed.dtype or current.device != seed.device
      or not current.is_contiguous()):
    raise ValueError("inverse FD sensor low seed does not match its producer")
  if current is not seed:
    current.copy_(seed)


def _inverse_fd_prepare_sensor_storage(sim, ns, seed=None):
  """Ensure inverseSkip can preserve skipped sensor stages transactionally."""
  if not ns:
    return
  sample = getattr(sim, "_sensordata", None)
  if sample is None:
    sample = torch.zeros((int(sim.batch_size), int(ns)), dtype=torch.float32,
                         device=sim.device)
    sim._sensordata = sample
  if seed is not None:
    if (not isinstance(seed, torch.Tensor) or tuple(seed.shape) != tuple(sample.shape)
        or seed.dtype != sample.dtype or seed.device != sample.device):
      raise ValueError("inverse FD sensor seed does not match sensordata")
    sample.copy_(seed)
  if (_inverse_fd_sensor_enabled(sim) and
      (getattr(sim, "_raw_sensor_program", None) is not None or
       getattr(sim, "_sensors", None) is not None)):
    raw = getattr(sim, "_raw_sensordata", None)
    if raw is None:
      sim._raw_sensordata = torch.zeros(
          (int(sim.batch_size), int(ns)), dtype=torch.float32,
          device=sim._state._qvel.device)


def _inverse_fd_query_qacc_low(sim, *, clone=True):
  """Capture only a current, provenance-matched acceleration low word."""
  provider = getattr(sim, "_qacc_low_for_sensor", None)
  if not callable(provider):
    return None
  value = provider(sim._state._qacc)
  if value is None:
    return None
  expected = tuple(sim._state._qacc.shape)
  if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected
      or value.dtype != torch.float32 or value.device != sim._state._qacc.device
      or not value.is_contiguous()):
    raise ValueError("inverse FD qacc low plane does not match qacc")
  return value.clone() if clone else value


def _inverse_constraint_pair_workspace_bytes(sim):
  """Charge paired row-force output and active twofold row scratch."""
  batch = int(sim.batch_size)
  nv = int(sim._mjmodel.nv)
  cc = getattr(sim, "_coupled_constraints", None)
  if cc is not None:
    nr = int(cc.descriptor.nr)
  else:
    legacy_rows = getattr(sim, "_legacy_canonical_rows", None)
    nr = (int(legacy_rows.shape[1])
          if getattr(legacy_rows, "ndim", 0) == 3 else 0)
  if nr < 0:
    raise ValueError("inverse constraint row count must be nonnegative")
  # Baseline and candidate each retain one additional constraint-force low
  # plane. The pair reducer also holds the row residual/force low vectors and
  # one bounded product/reduction plane beyond canonical-J storage.
  return batch * (2 * nv + 6 * nr) * 4


def _inverse_discrete_pair_workspace_bytes(sim):
  """Reserve transient paired operator/refinement vectors for inverseSkip."""
  import mujoco
  flags = int(getattr(sim._mjmodel.opt, "enableflags", 0))
  bit = int(getattr(mujoco.mjtEnableBit, "mjENBL_INVDISCRETE", 0))
  if not flags & bit:
    return 0
  batch = int(sim.batch_size)
  nv = int(sim._mjmodel.nv)
  # The dense path peaks at two RHS/product pairs plus the bounded refinement
  # pair. Component blocks stream per block row and fit within the same bound.
  return batch * max(nv, 1) * 32 * 4


def _inverse_fd_sensor_enabled(sim):
  """Match inverseSkip's sensor-disable gate for prepared prefix stages."""
  import mujoco
  disabled = int(getattr(sim._mjmodel.opt, "disableflags", 0)) & int(
      mujoco.mjtDisableBit.mjDSBL_SENSOR)
  return not bool(disabled)


def _inverse_fd_sensor_sample(sim, inverse_detail, ns, *, copy=True):
  """Return the inverseSkip-produced sample, optionally as its live view.

  `mjd_inverseFD` stages a zero-initialized output buffer before its center
  call and seeds it from the center sample before each perturbation. This
  preserves MuJoCo's stage-specific skip behavior and history reads even when
  Simulation had lazily left `_sensordata` unset.
  """
  sample = getattr(sim, "_sensordata", None)
  if sample is None:
    raise ValueError(
        "inverse FD sensor output has no allocated buffer or sensor producer")
  expected = (int(sim.batch_size), int(ns))
  device = sim._state._qvel.device
  if (not isinstance(sample, torch.Tensor) or tuple(sample.shape) != expected
      or sample.dtype != torch.float32
      or sample.device != device or not sample.is_contiguous()):
    raise ValueError("inverse FD sensor output does not match the model layout")
  return sample.clone() if copy else sample
