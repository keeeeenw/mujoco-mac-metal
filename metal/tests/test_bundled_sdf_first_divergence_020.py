# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Same-stage first-divergence diagnostic for bundled-SDF bowl/torus gates.

This corrects the stage-attribution flaws of ad-hoc rollout comparisons:

- Native step-stage quantities (status/time/qpos/qvel/qacc) are captured
  as detached clones BEFORE any recompute call can overwrite borrowed
  workspaces.
- Post-step assembly comparisons rerun CPU ``mj_forward`` at the matching
  represented native state (exact float64 copies of the native float32
  words) with identical applied forces, and are labeled as assembly
  evidence, separate from step-stage evidence.
- Contact rows use a validated bijection: unique nearest-position matching
  PLUS frame/normal alignment, distance/position tiers, row-span,
  active-set, dimension and multiplier-order checks.
- Stationarity/KKT evidence is reported scaled and absolute with solver
  route and used counts on both sides.

The trajectory gates themselves (019) own pass/fail on rollout equality;
this diagnostic owns methodology: its assertions pin the comparison
structure (bijection validity, stage/time identities, determinism,
assembly agreement tiers) and it REPORTS divergence magnitudes.
"""
import json
import math
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation
from test_bundled_sdf_rigid_oracle_019 import (
    _plugin_sphere_model,
    _cpu_contacts,
    _unique_native_slots,
)


def _clone(tensor):
  return np.asarray(tensor.detach().cpu().numpy()).copy()


def _native_step_stage(sim, qfrc):
  """Advance one native step; clone step-stage quantities immediately."""
  status = sim.step(1, qfrc_applied=qfrc)
  return {
      "stage": "native-step",
      "status": _clone(status).reshape(-1),
      "time": _clone(sim.state._time).reshape(-1),
      "qpos": _clone(sim.state._qpos),
      "qvel": _clone(sim.state._qvel),
      "qacc": _clone(sim.state._qacc),
  }


def _cpu_step_stage(data, qpos64, qvel64, qfrc64):
  """Advance one CPU step from exact float64 copies of a prior state.

  The caller owns `data` across steps so time is continuous (a true paired
  rollout); only the kinematic state is re-seated to the native
  represented state each step.
  """
  model = data.model
  data.qpos[:] = qpos64
  data.qvel[:] = qvel64
  data.qfrc_applied[:] = qfrc64
  mujoco.mj_step(model, data)
  return {
      "stage": "cpu-step",
      "status": np.zeros((1,), dtype=np.int32),
      "time": np.asarray([data.time], dtype=np.float64),
      "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
      "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
      "qacc": np.asarray(data.qacc, dtype=np.float64).copy(),
      "qfrc_constraint": np.asarray(
          data.qfrc_constraint, dtype=np.float64).copy(),
      "solver_niter": np.asarray(data.solver_niter).copy(),
  }


def _cpu_forward_at_state(model, qpos32, qvel32, qfrc):
  """CPU forward assembly at the matching represented native state."""
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(qpos32, dtype=np.float64)
  data.qvel[:] = np.asarray(qvel32, dtype=np.float64)
  data.qfrc_applied[:] = np.asarray(qfrc, dtype=np.float64).reshape(-1)
  mujoco.mj_forward(model, data)
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  return data, mass


def _native_assembly(sim, qfrc):
  """Native forward assembly at the current device state; clones first."""
  system = sim.assembled_system(recompute=True, qfrc_applied=qfrc)
  assert int(_clone(system["status"])[0]) == 0
  return {
      "stage": "native-assembly",
      "J": _clone(system["J"])[0],
      "R": _clone(system["R"])[0],
      "ar": _clone(system["ar"])[0],
      "rhs": _clone(system["rhs"])[0],
      "W": _clone(system["W"])[0],
      "lambda": _clone(system["lambda"])[0],
      "qacc": _clone(system["qacc"])[0],
      "qfrc_constraint": _clone(system["qfrc_constraint"])[0],
      "mass": _clone(system["mass_matrix"])[0],
      "status": _clone(system["status"]).reshape(-1),
      "mask": _clone(system["contact_mask"])[0],
      "pos": _clone(system["contact_position"])[0],
      "normal": _clone(system["contact_normal"])[0],
      "dist": _clone(system["contact_distance"])[0],
      "diagnostics": str(system.get("solver_diagnostics")),
  }


def _validate_bijection(model, sim, cpu_data, cpu_J, cpu_contacts, native):
  """Unique positional match plus frame/span/active/dimension/order proof."""
  assignment = _unique_native_slots(
      cpu_contacts, native["mask"], native["pos"])
  desc = sim._coupled_constraints.descriptor
  packed = np.asarray(
      desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  native_J = native["J"]
  native_lam = native["lambda"]
  matched_native_rows = []
  matched_cpu_rows = []
  row_report = []
  for ci, contact in enumerate(cpu_contacts):
    slot, position_error = assignment[ci]
    assert position_error < 1e-3, (ci, slot, position_error)
    # Frame correspondence: unit normals must align, not just positions.
    alignment = float(np.dot(contact["normal"], native["normal"][slot]))
    assert alignment > 0.999995, (ci, slot, alignment)
    np.testing.assert_allclose(
        native["dist"][slot], contact["dist"], rtol=2e-5, atol=5e-4)
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    gpu_rows = np.arange(gpu_base, gpu_base + len(contact["rows"]))
    source_rows = contact["rows"]
    # Dimension/active-set check: every CPU row is live on the native side.
    assert len(source_rows) == len(gpu_rows) == 1, (ci, contact["dim"])
    np.testing.assert_allclose(
        native_J[gpu_rows], cpu_J[source_rows], rtol=2e-5, atol=3e-6)
    matched_native_rows.extend(gpu_rows.tolist())
    matched_cpu_rows.extend(source_rows.tolist())
    row_report.append({
        "contact": ci, "slot": slot, "pos_err": position_error,
        "normal_alignment": alignment,
        "native_rows": gpu_rows.tolist(),
        "cpu_rows": source_rows.tolist(),
    })
  # Multiplier order check: same row order must carry the same multipliers.
  np.testing.assert_allclose(
      native_lam[np.asarray(matched_native_rows)],
      np.asarray(cpu_data.efc_force)[np.asarray(matched_cpu_rows)],
      rtol=2e-3, atol=1e-3)
  return row_report


def _stationarity(mass, qacc, qfrc_constraint, qfrc_smooth):
  residual = mass @ qacc - qfrc_constraint - qfrc_smooth
  scale = float(np.max(np.abs(mass @ qacc))) + 1e-30
  return {
      "abs_inf": float(np.max(np.abs(residual))),
      "scaled_inf": float(np.max(np.abs(residual))) / scale,
      "force_scale": scale,
  }


def _diagnose(name, steps=5):
  model = _plugin_sphere_model(name)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos0 = _clone(sim.state._qpos)[0].astype(np.float64)
  qvel0 = _clone(sim.state._qvel)[0].astype(np.float64)
  cpu_data, _, contacts0 = _cpu_contacts(model, qpos0, qvel0)
  assert len(contacts0) == 8 and all(c["dist"] < 0 for c in contacts0)
  dt = float(model.opt.timestep)
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  packet = {"fixture": name, "dt": dt, "steps": []}
  cpu_rollout = mujoco.MjData(model)
  # Determinism: repeated assembly at the same state is bitwise identical.
  first_assembly = _native_assembly(sim, qfrc)
  second_assembly = _native_assembly(sim, qfrc)
  for key in ("J", "R", "ar", "rhs", "W", "lambda", "qacc",
              "qfrc_constraint"):
    assert np.array_equal(first_assembly[key], second_assembly[key]), key
  for step in range(steps):
    # Paired rollout from the pre-step represented states.
    pre_qpos = _clone(sim.state._qpos)[0].astype(np.float64)
    pre_qvel = _clone(sim.state._qvel)[0].astype(np.float64)
    pre_time_native = float(_clone(sim.state._time).reshape(-1)[0])
    native = _native_step_stage(sim, qfrc)
    _cpu_step_stage(cpu_rollout, pre_qpos, pre_qvel, qfrc[0])
    cpu = {
        "stage": "cpu-step",
        "status": np.zeros((1,), dtype=np.int32),
        "time": np.asarray([cpu_rollout.time], dtype=np.float64),
        "qpos": np.asarray(cpu_rollout.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(cpu_rollout.qvel, dtype=np.float64).copy(),
        "qacc": np.asarray(cpu_rollout.qacc, dtype=np.float64).copy(),
        "qfrc_constraint": np.asarray(
            cpu_rollout.qfrc_constraint, dtype=np.float64).copy(),
        "solver_niter": np.asarray(cpu_rollout.solver_niter).copy(),
    }
    # Stage/time identities: both sides advanced exactly one dt.
    assert native["status"].tolist() == [0]
    assert abs(float(native["time"][0]) - (pre_time_native + dt)) < 1e-9
    assert abs(float(cpu["time"][0]) - (pre_time_native + dt)) < 1e-6
    step_entry = {
        "step": step,
        "native_time": float(native["time"][0]),
        "cpu_time": float(cpu["time"][0]),
        "qpos_err": float(np.max(np.abs(native["qpos"][0] - cpu["qpos"]))),
        "qvel_err": float(np.max(np.abs(native["qvel"][0] - cpu["qvel"]))),
        "qacc_err": float(np.max(np.abs(native["qacc"][0] - cpu["qacc"]))),
        "cpu_solver_niter": [int(v) for v in cpu["solver_niter"][:4]],
    }
    # Same-state assembly at the post-step native state.
    post_qpos = native["qpos"][0]
    post_qvel = native["qvel"][0]
    assembly = _native_assembly(sim, qfrc)
    cpu_fwd, cpu_mass = _cpu_forward_at_state(
        model, post_qpos, post_qvel, qfrc[0])
    _, cpu_J, cpu_contacts = _cpu_contacts_from_data(model, cpu_fwd)
    rows = _validate_bijection(
        model, sim, cpu_fwd, cpu_J, cpu_contacts, assembly)
    cpu_smooth = (np.asarray(cpu_fwd.qfrc_bias, dtype=np.float64)
                  + np.asarray(qfrc[0], dtype=np.float64))
    cpu_stat = _stationarity(
        cpu_mass, np.asarray(cpu_fwd.qacc), np.asarray(
            cpu_fwd.qfrc_constraint), cpu_smooth)
    native_stat = _stationarity(
        assembly["mass"], assembly["qacc"],
        assembly["qfrc_constraint"], cpu_smooth)
    step_entry["assembly"] = {
        "rows": rows,
        "cpu_stationarity": cpu_stat,
        "native_stationarity": native_stat,
        "native_diagnostics": assembly["diagnostics"][:200],
    }
    packet["steps"].append(step_entry)
  print(f"FIRST_DIVERGENCE_PACKET[{name}] "
        + json.dumps(packet, default=float))
  return packet


def _cpu_contacts_from_data(model, cpu_data):
  cpu_J = np.asarray(cpu_data.efc_J).reshape(cpu_data.nefc, model.nv)
  contacts = []
  for i in range(cpu_data.ncon):
    contact = cpu_data.contact[i]
    count = 1 if int(contact.dim) == 1 else (
        2 * (int(contact.dim) - 1)
        if int(model.opt.cone) == 0 else int(contact.dim))
    base = int(contact.efc_address)
    contacts.append({
        "dist": float(contact.dist),
        "normal": np.asarray(contact.frame[:3], dtype=np.float64).copy(),
        "pos": np.asarray(contact.pos, dtype=np.float64).copy(),
        "rows": np.arange(base, base + count, dtype=np.int64),
    })
  return cpu_data, cpu_J, contacts


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in same-stage first-divergence diagnostic")
@pytest.mark.parametrize("name", ["torus", "bowl"])
def test_same_stage_first_divergence_packet_gpu(name):
  packet = _diagnose(name)
  assert len(packet["steps"]) == 5
  for entry in packet["steps"]:
    assert entry["assembly"]["rows"] and len(
        entry["assembly"]["rows"]) == 8
    # Stage identities are explicit and consistent on every step.
    assert math.isfinite(entry["native_time"])
    assert abs(entry["native_time"] - entry["cpu_time"]) < 1e-6
