# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned native counterpart of MuJoCo 3.10 ``mj_transmission``.

The query reuses the simulation's production POS-stage actuator kinematics.
It returns dense moment rows, while upstream MuJoCo stores the same rows in
packed ``moment_rownnz``/``moment_rowadr``/``moment_colind`` form. It does not
compute actuator velocity (that belongs to the later velocity stage). The
restricted scalar-motor profile uses its compiled direct-joint map, matching
the same position-stage length and moment convention.
"""

import torch
import numpy as np

_I32_MAX = (1 << 31) - 1


def _checked_counts(batch, nu, nv):
  counts = {
      "batch": batch,
      "actuator count": nu,
      "DOF count": nv,
      "actuator length output": batch * nu,
      "dense actuator moment output": batch * nu * nv,
  }
  for name, value in counts.items():
    if value < 0 or value > _I32_MAX:
      raise ValueError(f"transmission query {name} exceeds int32 indexing")
  return counts


def _preflight(sim, batch, nu, nv, scalar_program, scalar_motor=None):
  """Admit rollback storage, final outputs, and query-local temporaries."""
  from mujoco_metal.finite_difference import estimate_transaction_bytes

  counts = _checked_counts(batch, nu, nv)
  output_count = counts["actuator length output"] + counts[
      "dense actuator moment output"] + batch
  # Finite checks reduce one dense output-sized bool buffer at a time; status
  # combination keeps a few batch-sized masks alive through output cloning.
  finite_count = counts["dense actuator moment output"] + batch * nu + 4 * batch
  if output_count > _I32_MAX or finite_count > _I32_MAX:
    raise ValueError("transmission query temporary storage exceeds int32 indexing")
  extra = 4 * output_count + finite_count
  if scalar_program is not None and scalar_program._position_length is None:
    # capture_position_context lazily owns this output scratch.
    extra += 4 * batch * max(nu, 1)
  if scalar_motor is not None:
    # This restricted profile has no general POS transmission program. Its
    # direct-joint qpos addresses and dense moment map are static model data,
    # materialized as bounded MPS query scratch here.
    extra += 8 * nu + 4 * nu * nv + 4 * nu + 4 * batch * nu
  if extra > _I32_MAX * 4:
    raise ValueError("transmission query byte count exceeds int32 addressing")
  # The existing prepared record is cloned by the rollback transaction while
  # prepare_forward_position may retain a second set of owned POS context
  # tensors until that transaction restores the old record. Charge a second
  # full transaction-storage bound for this transient overlap; it also covers
  # bounded plugin snapshot payloads and is conservative when contexts reuse
  # preallocated owner buffers.
  snapshot_bound = estimate_transaction_bytes(sim)
  required = 2 * snapshot_bound + extra
  if required > _I32_MAX * 4:
    raise ValueError("transmission query transaction exceeds int32 byte addressing")
  budget = int(getattr(getattr(sim, "limits", None),
                       "memory_budget_bytes", 1 << 30))
  if required > budget:
    raise ValueError(
        f"transmission query needs {required} bytes; memory budget is {budget}")
  return required


def _validate_output(value, shape, name, device):
  if (not isinstance(value, torch.Tensor) or value.device.type != "mps"
      or (device.index is not None and value.device.index != device.index)
      or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape)
      or not value.is_contiguous()):
    raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
  return value


def _scalar_motor_static_map(model, motor_model):
  """Validate and lower the scalar profile's static direct-joint map."""
  nu, nv = int(model.nu), int(model.nv)
  joint_ids = np.asarray(model.actuator_trnid[:nu, 0], dtype=np.int64)
  if (joint_ids.shape != (nu,) or np.any(joint_ids < 0)
      or np.any(joint_ids >= int(model.njnt))):
    raise RuntimeError("scalar motor actuator joint map is invalid")
  qpos_address = np.asarray(model.jnt_qposadr, dtype=np.int64)[joint_ids]
  dof = np.asarray(motor_model.dof, dtype=np.int64)
  gear = np.asarray(motor_model.gear, dtype=np.float32)
  if (qpos_address.shape != (nu,) or np.any(qpos_address < 0)
      or np.any(qpos_address >= int(model.nq))
      or dof.shape != (nu,) or np.any(dof < 0) or np.any(dof >= nv)
      or gear.shape != (nu,) or not np.all(np.isfinite(gear))):
    raise RuntimeError("scalar motor transmission addresses are invalid")
  dense = np.zeros((nu, nv), dtype=np.float32)
  dense[np.arange(nu), dof] = gear
  return np.ascontiguousarray(qpos_address), np.ascontiguousarray(gear), dense


def _finish_outputs(length, moment, status):
  """Own outputs, promote nonfinite rows, and isolate failed worlds."""
  b = int(length.shape[0])
  finite = torch.ones((b,), dtype=torch.bool, device=length.device)
  for value in (length, moment):
    if value.numel():
      finite = finite & torch.isfinite(value).flatten(1).all(dim=1)
  status = torch.where((status == 0) & ~finite,
                       torch.full_like(status, 3), status)
  failed = status != 0
  out_length = length.clone()
  out_moment = moment.clone()
  out_length.masked_fill_(failed[:, None], 0.0)
  out_moment.masked_fill_(failed[:, None, None], 0.0)
  return {"actuator_length": out_length,
          "actuator_moment": out_moment,
          "status": status.clone()}


def mj_transmission(sim):
  """Return current-state actuator lengths, dense moments, and row status.

  All numerical kinematics run through the native POS program on MPS. The
  result owns its storage and the query restores the simulation's prepared
  records, caches, workspaces, and plugin state on success or exception.
  ``actuator_moment`` has shape ``[B,nu,nv]``; its dense representation is
  equivalent to expanding upstream MuJoCo's packed moment rows.
  """
  model = sim.model
  batch, nu, nv = int(sim.batch_size), int(model.nu), int(model.nv)
  _checked_counts(batch, nu, nv)
  actuators = getattr(sim, "_actuators", None)
  scalar = getattr(sim, "_transmissions", None)
  scalar_motor = getattr(sim, "_motor", None)
  if nu and actuators is None and scalar is None and scalar_motor is None:
    raise NotImplementedError(
        "the active simulation profile has no admitted transmission stage")
  _preflight(sim, batch, nu, nv,
             scalar if nu and actuators is None else None,
             scalar_motor if nu and actuators is None and scalar is None else None)
  device = sim.state._device
  from mujoco_metal.native_api import _inverse_query_workspaces
  with _inverse_query_workspaces(sim):
    if nu == 0:
      length = torch.empty((batch, 0), dtype=torch.float32, device=device)
      moment = torch.empty((batch, 0, nv), dtype=torch.float32, device=device)
      status = torch.zeros((batch,), dtype=torch.int32, device=device)
      return _finish_outputs(length, moment, status)

    from mujoco_metal.forward_stages import ForwardStage
    record = sim.prepare_forward_position(skipsensor=True)
    position = sim.validate_forward_stage_record(record, ForwardStage.POS)
    values = position.values[ForwardStage.POS]
    context = values.get("context", {})
    dynamics = values.get("dynamics", {})
    status = dynamics.get("pose_status")
    if status is None:
      status = torch.zeros((batch,), dtype=torch.int32, device=device)
    elif (not isinstance(status, torch.Tensor)
          or tuple(status.shape) != (batch,) or status.dtype != torch.int32
          or status.device.type != "mps" or not status.is_contiguous()):
      raise ValueError("position stage returned invalid per-world status")

    if actuators is not None:
      kin = context.get("actuators")
      if not isinstance(kin, dict):
        raise RuntimeError("POS stage did not publish actuator kinematics")
      length = _validate_output(
          kin.get("length"), (batch, nu), "actuator length", device)
      packed_dense = _validate_output(
          kin.get("moment"), (batch, nu, max(nv, 1)),
          "actuator moment workspace", device)
      moment = packed_dense[:, :, :nv]
    elif scalar is not None:
      transmission = context.get("transmissions")
      if not isinstance(transmission, dict):
        raise RuntimeError("POS stage did not publish transmission position context")
      length = _validate_output(
          transmission.get("length"), (batch, max(nu, 1)),
          "transmission position length", device)[:, :nu]
      moment_map = scalar._moment_map
      if (not isinstance(moment_map, torch.Tensor)
          or moment_map.dtype != torch.float32
          or moment_map.device.type != "mps"):
        raise ValueError("scalar transmission moment map has invalid shape")
      if nv:
        if moment_map.numel() != nu * nv:
          raise ValueError("scalar transmission moment map has invalid shape")
        moment = moment_map.reshape(nu, nv).unsqueeze(0).expand(batch, nu, nv)
      else:
        moment = torch.empty((batch, nu, 0), dtype=torch.float32,
                             device=device)
    else:
      # ScalarMotorModel admits only fixed-gain direct hinge/slide joint
      # transmissions. MuJoCo's current transmission length is gear*qpos and
      # its moment row has one gear entry at the joint DOF.
      model = sim.model
      qpos_address, gear, dense = _scalar_motor_static_map(
          model, scalar_motor.model)
      qpos_indices = torch.as_tensor(
          qpos_address.copy(), dtype=torch.int64, device=device)
      gear_device = torch.as_tensor(gear.copy(), dtype=torch.float32,
                                    device=device)
      length = sim.state._qpos.index_select(1, qpos_indices) * gear_device
      moment_map = torch.as_tensor(dense, dtype=torch.float32, device=device)
      moment = moment_map.unsqueeze(0).expand(batch, nu, nv)

    return _finish_outputs(length, moment, status)
