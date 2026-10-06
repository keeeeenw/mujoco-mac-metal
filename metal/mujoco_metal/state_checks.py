# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native device equivalents of MuJoCo 3.10 checkPos/checkVel/checkAcc."""

import mujoco

_MAXVAL = 1.0e10


def _awake_dof_mask(sim):
  """Return the source-ordered awake DOF mask, or None without sleep."""
  torch = sim.state._torch
  scheduler = getattr(sim, "_sleep_schedule", None)
  nv = int(sim.model.nv)
  if scheduler is None or nv == 0:
    return None
  tree_ids = sim._sleep_dof_treeid
  safe_tree_ids = tree_ids.clamp_min(0)
  awake = scheduler.tree_awake.index_select(1, safe_tree_ids) != 0
  # dof_treeid is compiled static metadata; MuJoCo's awake list excludes
  # DOFs with no dynamic tree as well.
  return awake & (tree_ids[None, :] >= 0)


def _bad_values(values, active=None, *, torch):
  batch, width = int(values.shape[0]), int(values.shape[1])
  if width == 0:
    return (torch.zeros((batch,), dtype=torch.bool, device=values.device),
            torch.zeros((batch,), dtype=torch.int32, device=values.device))
  bad = (~torch.isfinite(values)) | (values.abs() > _MAXVAL)
  if active is not None:
    bad = bad & active
  present = bad.any(dim=1)
  index = torch.argmax(bad.to(torch.int32), dim=1).to(torch.int32)
  return present, index


def _record_warning(sim, warning, bad_world, bad_index):
  state = sim.state
  torch = state._torch
  auto_reset = not bool(
      int(sim.model.opt.disableflags)
      & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET))
  if hasattr(state, "record_warning_rows"):
    if auto_reset:
      sim._reset_from_check(
          bad_world,
          forward_acc=(warning == int(mujoco.mjtWarning.mjWARN_BADQACC)),
          warning_commit=lambda: state.record_warning_rows(
              bad_world, bad_index, warning, autoreset=True))
    else:
      state.record_warning_rows(
          bad_world, bad_index, warning, autoreset=False)
    return {
        "bad_world": bad_world.clone(),
        "first_bad_index": torch.where(
            bad_world, bad_index, torch.full_like(bad_index, -1)).clone(),
        "warning_number": state._warning_number[:, warning].clone(),
        "warning_lastinfo": state._warning_lastinfo[:, warning].clone(),
    }
  number = state._warning_number.clone()
  lastinfo = state._warning_lastinfo.clone()
  # Pinned mj_check* raises the warning before AUTO-reset. Reset clears the
  # warning table, then the check installs the post-reset count/info before
  # mj_checkAcc's recovery forward begins.
  number[:, warning] = torch.where(
      bad_world, number[:, warning] + 1, number[:, warning])
  lastinfo[:, warning] = torch.where(
      bad_world, bad_index, lastinfo[:, warning])
  state._warning_number.copy_(number)
  state._warning_lastinfo.copy_(lastinfo)
  # mj_check* calls mj_warning (+1), resets warning stats when autorecovery is
  # enabled, then explicitly increments (+1). mj_resetData clears the row, so
  # the observable post-reset count is one; without reset it is two per call.
  if auto_reset:
    def commit_warning_after_reset():
      final_number = torch.where(
          bad_world[:, None], torch.zeros_like(state._warning_number),
          state._warning_number)
      final_info = torch.where(
          bad_world[:, None], torch.zeros_like(state._warning_lastinfo),
          state._warning_lastinfo)
      final_number[:, warning] = torch.where(
          bad_world, torch.ones_like(bad_index), final_number[:, warning])
      final_info[:, warning] = torch.where(
          bad_world, bad_index, final_info[:, warning])
      state._warning_number.copy_(final_number)
      state._warning_lastinfo.copy_(final_info)

    sim._reset_from_check(
        bad_world,
        forward_acc=(warning == int(mujoco.mjtWarning.mjWARN_BADQACC)),
        warning_commit=commit_warning_after_reset)
  else:
    # Without AUTORESET MuJoCo records both the generic warning and the
    # explicit bad-state warning.
    number[:, warning] = torch.where(
        bad_world, number[:, warning] + 1, number[:, warning])
    state._warning_number.copy_(number)
  return {
      "bad_world": bad_world.clone(),
      "first_bad_index": torch.where(
          bad_world, bad_index, torch.full_like(bad_index, -1)).clone(),
      "warning_number": state._warning_number[:, warning].clone(),
      "warning_lastinfo": state._warning_lastinfo[:, warning].clone(),
  }


def mj_checkPos(sim):
  """Check qpos, record the first bad index, and apply source AUTORESET."""
  bad, index = _bad_values(sim.state._qpos, torch=sim.state._torch)
  return _record_warning(
      sim, int(mujoco.mjtWarning.mjWARN_BADQPOS), bad, index)


def mj_checkVel(sim):
  """Check qvel, filtering to awake DOFs when MuJoCo sleep is active."""
  active = _awake_dof_mask(sim)
  bad, index = _bad_values(sim.state._qvel, active, torch=sim.state._torch)
  return _record_warning(
      sim, int(mujoco.mjtWarning.mjWARN_BADQVEL), bad, index)


def mj_checkAcc(sim, *, qacc=None):
  """Check qacc and source-reset/forward bad worlds when enabled."""
  active = _awake_dof_mask(sim)
  torch = sim.state._torch
  values = sim.state._qacc if qacc is None else qacc
  if (not isinstance(values, torch.Tensor)
      or values.dtype != torch.float32
      or tuple(values.shape) != tuple(sim.state._qacc.shape)
      or values.device != sim.state._qacc.device
      or not values.is_contiguous()):
    raise ValueError("qacc must be contiguous float32 on the state device")
  bad, index = _bad_values(values, active, torch=torch)
  return _record_warning(
      sim, int(mujoco.mjtWarning.mjWARN_BADQACC), bad, index)
