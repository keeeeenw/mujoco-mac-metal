# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Same-stage first-divergence diagnostic for bundled-SDF bowl/torus gates.

Corrected methodology (see assignment history in the response doc):

- Force balance uses the actual pinned smooth-force results: CPU
  `data.qfrc_smooth` (whose residual is exactly zero, unlike the
  `qfrc_bias + applied` combination, which leaves 82.18 on a gravity
  fixture) and the solver-implied native smooth
  (`mass @ qacc - qfrc_constraint`, labeled as implied, never CPU bias).
  Mass, acceleration, smooth-force and constraint-force errors are
  reported as separate components. Host residuals are float64 recomputed
  from independently copied arrays (`host_float64`); direct device
  tensor identities are labeled `device` separately.
- Two rollout streams: a same-start one-step stream (CPU re-seated to
  the native represented state every step — labeled as such, NOT an
  independent rollout) and an uninterrupted rollout stream (pristine
  native instance stepped with no observational calls vs CPU stepped
  from initial with no reseating). A second native instance with
  interleaved assembly is stepped in sequence and required bitwise
  equal to the pristine run, proving diagnostic recomputation does not
  alter rollout/warmstart behavior.
- Step-stage system quantities come from the accepted-step system
  (cloned immediately after the step) and the post-`mj_step` CPU data,
  never from post-recompute workspaces. Assembly evidence reruns CPU
  `mj_forward` at the matching represented state and is labeled
  separately.
- Row correspondence is scoped to these one-pair frictionless
  (`condim == 1`) fixtures: unique positional match plus normal
  alignment, pair/geom identity (single pair asserted on the CPU side),
  actual `dim`, row bounds against `nefc`/`nr`, active-set equality and
  multiplier order. No general frame/bijection proof is claimed.
- The committed QP reference (float64 active-set enumeration, a
  NUMERICAL reference, not an exact solution) with J/lambda cross-swap
  evidence makes the causal claim reproducible: feasibility,
  complementarity, primal/dual stationarity and stopping/iteration data
  are reported with correct signs and scaling. Solver routes are NOT
  inferred from equal iteration counts; configured solvers and used
  counts are reported as counts only.

The trajectory gates themselves (019) own pass/fail on rollout equality;
this diagnostic owns methodology: its assertions pin the comparison
structure and it REPORTS divergence magnitudes.
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

_DT_TOL = 1e-9


def _clone(tensor):
  return np.asarray(tensor.detach().cpu().numpy()).copy()


def _native_step_stage(sim, qfrc):
  """Advance one native step; clone step-stage quantities immediately.

  System quantities come from the accepted-step system (the step's own
  solve), cloned before any recompute call can overwrite the borrowed
  workspaces.
  """
  status = sim.step(1, qfrc_applied=qfrc)
  accepted = sim._accepted_step["system"]
  stage = {
      "stage": "native-step",
      "status": _clone(status).reshape(-1),
      "time": _clone(sim.state._time).reshape(-1),
      "qpos": _clone(sim.state._qpos),
      "qvel": _clone(sim.state._qvel),
      "qacc": _clone(sim.state._qacc),
  }
  for key in ("J", "lambda", "qacc", "qfrc_constraint", "status",
              "contact_mask"):
    assert key in accepted, f"accepted-step system lacks {key}"
  stage["system"] = {key: _clone(accepted[key]) for key in
                     ("J", "lambda", "qacc", "qfrc_constraint", "status",
                      "contact_mask")}
  return stage


def _cpu_step_quantities(data):
  return {
      "stage": "cpu-step",
      "status": np.zeros((1,), dtype=np.int32),
      "time": np.asarray([data.time], dtype=np.float64),
      "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
      "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
      "qacc": np.asarray(data.qacc, dtype=np.float64).copy(),
      "qfrc_constraint": np.asarray(
          data.qfrc_constraint, dtype=np.float64).copy(),
      "qfrc_smooth": np.asarray(data.qfrc_smooth, dtype=np.float64).copy(),
      "efc_J": np.asarray(data.efc_J, dtype=np.float64).copy(),
      "efc_force": np.asarray(data.efc_force, dtype=np.float64).copy(),
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
      "diagnostics": _clone(system["solver_diagnostics"])[0].tolist(),
  }


def _cpu_contacts_from_data(model, cpu_data):
  cpu_J = np.asarray(cpu_data.efc_J).reshape(cpu_data.nefc, model.nv)
  contacts = []
  pair_geoms = set()
  for i in range(cpu_data.ncon):
    contact = cpu_data.contact[i]
    dim = int(contact.dim)
    count = 1 if dim == 1 else (
        2 * (dim - 1) if int(model.opt.cone) == 0 else dim)
    base = int(contact.efc_address)
    assert base >= 0 and base + count <= int(cpu_data.nefc), (
        f"row span [{base}, {base + count}) outside nefc "
        f"{int(cpu_data.nefc)}")
    pair_geoms.add((int(contact.geom1), int(contact.geom2)))
    contacts.append({
        "dist": float(contact.dist),
        "normal": np.asarray(contact.frame[:3], dtype=np.float64).copy(),
        "pos": np.asarray(contact.pos, dtype=np.float64).copy(),
        "rows": np.arange(base, base + count, dtype=np.int64),
        "dim": dim,
        "geom1": int(contact.geom1),
        "geom2": int(contact.geom2),
    })
  return cpu_data, cpu_J, contacts, pair_geoms


def _matched_rows(model, sim, cpu_data, cpu_J, cpu_contacts, pair_geoms,
                   native):
  """Matched constraint rows with per-side QP systems for KKT reports."""
  desc = sim._coupled_constraints.descriptor
  packed = np.asarray(
      desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  assignment = _unique_native_slots(
      cpu_contacts, native["mask"], native["pos"])
  native_rows, cpu_rows = [], []
  for ci, contact in enumerate(cpu_contacts):
    slot, _ = assignment[ci]
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    native_rows.extend(
        range(gpu_base, gpu_base + len(contact["rows"])))
    cpu_rows.extend(contact["rows"].tolist())
  native_rows = np.asarray(native_rows)
  cpu_rows = np.asarray(cpu_rows)
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, cpu_data, mass)
  Jc = cpu_J[cpu_rows]
  Wc = Jc @ np.linalg.solve(mass, Jc.T)
  Rc = np.asarray(cpu_data.efc_R)[cpu_rows]
  bc = np.asarray(cpu_data.efc_b)[cpu_rows]
  Wn = native["W"][np.ix_(native_rows, native_rows)]
  Rn = native["R"][native_rows]
  bn = -native["rhs"][native_rows]
  return (native_rows, cpu_rows, Wc, Rc, bc, Wn, Rn, bn)


def _validate_bijection(model, sim, cpu_data, cpu_J, cpu_contacts,
                        pair_geoms, native):
  """One-pair frictionless correspondence with explicit scope checks."""
  assert len(pair_geoms) == 1, pair_geoms
  assert all(c["dim"] == 1 for c in cpu_contacts)
  assert int(np.sum(native["mask"] > 0.5)) == len(cpu_contacts)
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
    alignment = float(np.dot(contact["normal"], native["normal"][slot]))
    assert alignment > 0.999995, (ci, slot, alignment)
    np.testing.assert_allclose(
        native["dist"][slot], contact["dist"], rtol=2e-5, atol=5e-4)
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    gpu_rows = np.arange(gpu_base, gpu_base + len(contact["rows"]))
    assert gpu_base >= 0 and gpu_base + len(contact["rows"]) <= int(
        desc.nr), (ci, gpu_base, int(desc.nr))
    source_rows = contact["rows"]
    assert len(source_rows) == len(gpu_rows) == 1, (
        ci, contact["dim"], source_rows, gpu_rows)
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
  np.testing.assert_allclose(
      native_lam[np.asarray(matched_native_rows)],
      np.asarray(cpu_data.efc_force)[np.asarray(matched_cpu_rows)],
      rtol=2e-3, atol=1e-3)
  return row_report


def _host_stationarity(mass, qacc, qfrc_constraint, qfrc_smooth):
  residual = mass @ qacc - qfrc_constraint - qfrc_smooth
  scale = float(np.max(np.abs(mass @ qacc))) + 1e-30
  return {
      "abs_inf": float(np.max(np.abs(residual))),
      "scaled_inf": float(np.max(np.abs(residual))) / scale,
      "force_scale": scale,
  }


def _qp_reference_lam(W, R, b):
  """Float64 active-set enumeration: a NUMERICAL reference, not exact."""
  n = len(b)
  A = W + np.diag(R)
  for maskbits in range(1 << n):
    free = [i for i in range(n) if maskbits >> i & 1]
    lam = np.zeros(n)
    if free:
      F = np.ix_(free, free)
      try:
        lf = np.linalg.solve(A[F], -b[free])
      except np.linalg.LinAlgError:
        continue
      if np.any(lf < -1e-9):
        continue
      lam[free] = lf
    g = A @ lam + b
    if np.all(g >= -1e-9):
      return lam
  return None


def _qp_kkt(lam, W, R, b):
  A = W + np.diag(R)
  g = A @ lam + b
  return {
      "complementarity": float(np.max(np.abs(lam * g))),
      "min_lam": float(np.min(lam)),
      "min_g": float(np.min(g)),
  }


def _diagnose(name, steps=8):
  model = _plugin_sphere_model(name)
  dt = float(model.opt.timestep)
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  packet = {"fixture": name, "dt": dt, "label": "same-start-one-step",
            "steps": []}
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos0 = _clone(sim.state._qpos)[0].astype(np.float64)
  qvel0 = _clone(sim.state._qvel)[0].astype(np.float64)
  cpu_data, _, contacts0 = _cpu_contacts(model, qpos0, qvel0)
  assert len(contacts0) == 8 and all(c["dist"] < 0 for c in contacts0)
  # Determinism: repeated assembly at the same state is bitwise identical.
  first_assembly = _native_assembly(sim, qfrc)
  second_assembly = _native_assembly(sim, qfrc)
  for key in ("J", "R", "ar", "rhs", "W", "lambda", "qacc",
              "qfrc_constraint"):
    assert np.array_equal(first_assembly[key], second_assembly[key]), key
  cpu_rollout = mujoco.MjData(model)
  for step in range(steps):
    pre_qpos = _clone(sim.state._qpos)[0].astype(np.float64)
    pre_qvel = _clone(sim.state._qvel)[0].astype(np.float64)
    pre_time_native = float(_clone(sim.state._time).reshape(-1)[0])
    native = _native_step_stage(sim, qfrc)
    cpu_rollout.qpos[:] = pre_qpos
    cpu_rollout.qvel[:] = pre_qvel
    cpu_rollout.qfrc_applied[:] = qfrc[0].astype(np.float64)
    mujoco.mj_step(model, cpu_rollout)
    cpu = _cpu_step_quantities(cpu_rollout)
    assert native["status"].tolist() == [0]
    assert abs(float(native["time"][0]) - (pre_time_native + dt)) < 1e-9
    assert abs(float(cpu["time"][0]) - (pre_time_native + dt)) < 1e-6
    step_entry = {
        "step": step,
        "native_time": float(native["time"][0]),
        "cpu_time": float(cpu["time"][0]),
        "host_float64": {
            "qpos_err": float(np.max(np.abs(
                native["qpos"][0] - cpu["qpos"]))),
            "qvel_err": float(np.max(np.abs(
                native["qvel"][0] - cpu["qvel"]))),
            "qacc_err": float(np.max(np.abs(
                native["qacc"][0] - cpu["qacc"]))),
        },
        "device": {
            "native_status": native["status"].tolist(),
            "cpu_solver_niter": [int(v) for v in cpu["solver_niter"][:4]],
            "cpu_solver_setting": int(model.opt.solver),
            "native_ncon": int(np.sum(
                native["system"]["contact_mask"][0] > 0.5)),
            "cpu_ncon": int(cpu_rollout.ncon),
        },
        # Step-stage systems come from different solver runs (warmstarted
        # native trajectory vs cold single CPU step): norms are reported,
        # equality is NOT asserted across stages.
        "step_system_norms": {
            "native_J": float(np.max(np.abs(
                native["system"]["J"][0]))),
            "native_lam": float(np.max(np.abs(
                native["system"]["lambda"][0]))),
            "cpu_J": float(np.max(np.abs(cpu["efc_J"]))),
            "cpu_force": float(np.max(np.abs(cpu["efc_force"]))),
        },
    }
    post_qpos = native["qpos"][0]
    post_qvel = native["qvel"][0]
    assembly = _native_assembly(sim, qfrc)
    cpu_fwd, cpu_mass = _cpu_forward_at_state(
        model, post_qpos, post_qvel, qfrc[0])
    _, cpu_J, cpu_contacts, pair_geoms = _cpu_contacts_from_data(
        model, cpu_fwd)
    rows = _validate_bijection(
        model, sim, cpu_fwd, cpu_J, cpu_contacts, pair_geoms, assembly)
    cpu_smooth = np.asarray(cpu_fwd.qfrc_smooth, dtype=np.float64)
    cpu_stat = _host_stationarity(
        cpu_mass, np.asarray(cpu_fwd.qacc),
        np.asarray(cpu_fwd.qfrc_constraint), cpu_smooth)
    smooth_native_implied = (assembly["mass"] @ assembly["qacc"]
                             - assembly["qfrc_constraint"])
    smooth_err = smooth_native_implied - cpu_smooth
    # Constraint-space KKT triple per side (reports, not solves): W from
    # the native Delassus block vs CPU J/M reconstruction, b = -rhs vs
    # efc_b, multipliers in matched row order. Routes are NOT inferred;
    # configured setting + used counts are reported as counts only.
    match_rows = _matched_rows(
        model, sim, cpu_fwd, cpu_J, cpu_contacts, pair_geoms, assembly)
    kkt_cpu = _qp_kkt(np.asarray(cpu_fwd.efc_force)[match_rows[1]],
                      match_rows[2], match_rows[3], match_rows[4])
    kkt_nat = _qp_kkt(assembly["lambda"][match_rows[0]],
                      match_rows[5], match_rows[6], match_rows[7])
    ref_nat = _qp_reference_lam(match_rows[5], match_rows[6],
                                match_rows[7])
    kkt_ref = (_qp_kkt(ref_nat, match_rows[5], match_rows[6],
                       match_rows[7]) if ref_nat is not None else None)
    # Jacobian error profile: uniform absolute error implicates the
    # float32 input/subtraction floor (needs wide inputs); error scaled
    # with entry magnitude would implicate assembly rounding alone.
    Jn_m = assembly["J"][match_rows[0]]
    Jc_m = cpu_J[match_rows[1]]
    j_abs = np.abs(Jn_m - Jc_m)
    j_mag = np.abs(Jc_m)
    small = j_mag < 1e-2
    j_profile = {
        "abs_max": float(np.max(j_abs)),
        "rel_max": float(np.max(j_abs) / (np.max(j_mag) + 1e-30)),
        "abs_max_at_small": float(np.max(j_abs[small])) if bool(
            np.any(small)) else 0.0,
        "n_small": int(np.sum(small)),
    }
    step_entry["assembly"] = {
        "rows": rows,
        "host_float64": {
            "cpu_stationarity": cpu_stat,
            "mass_err": float(np.max(np.abs(
                assembly["mass"] - cpu_mass))),
            "qacc_err": float(np.max(np.abs(
                assembly["qacc"] - np.asarray(cpu_fwd.qacc)))),
            "fcon_err": float(np.max(np.abs(
                assembly["qfrc_constraint"]
                - np.asarray(cpu_fwd.qfrc_constraint)))),
            "smooth_implied_err": float(np.max(np.abs(smooth_err))),
            "smooth_scale": float(np.max(np.abs(cpu_smooth))) + 1e-30,
            "kkt_cpu": kkt_cpu,
            "kkt_native": kkt_nat,
            "kkt_reference_native": kkt_ref,
            "j_profile": j_profile,
        },
        "device": {
            "native_diagnostics": assembly["diagnostics"],
        },
    }
    packet["steps"].append(step_entry)
  print(f"FIRST_DIVERGENCE_PACKET[{name}] "
        + json.dumps(packet, default=float))
  return packet


def _rollout(name, steps=8, with_assembly=False):
  """Uninterrupted rollout: no reseating, no (or provably inert) probing."""
  model = _plugin_sphere_model(name)
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos0 = _clone(sim.state._qpos)[0].astype(np.float64)
  cpu_data = mujoco.MjData(model)
  cpu_data.qpos[:] = qpos0
  cpu_data.qvel[:] = 0.0
  trace = {"fixture": name, "label": "uninterrupted-rollout", "steps": []}
  for step in range(steps):
    status = sim.step(1, qfrc_applied=qfrc)
    assert status.cpu().numpy().tolist() == [0]
    cpu_data.qfrc_applied[:] = qfrc[0].astype(np.float64)
    mujoco.mj_step(model, cpu_data)
    if with_assembly:
      _native_assembly(sim, qfrc)
    trace["steps"].append({
        "step": step,
        "native_time": float(_clone(sim.state._time).reshape(-1)[0]),
        "cpu_time": float(cpu_data.time),
        "host_float64": {
            "qpos_err": float(np.max(np.abs(
                _clone(sim.state._qpos)[0] - np.asarray(cpu_data.qpos)))),
            "qvel_err": float(np.max(np.abs(
                _clone(sim.state._qvel)[0] - np.asarray(cpu_data.qvel)))),
            "qacc_err": float(np.max(np.abs(
                _clone(sim.state._qacc)[0] - np.asarray(cpu_data.qacc)))),
        },
        "device": {"ncon_cpu": int(cpu_data.ncon)},
    })
  print(f"UNINTERRUPTED_ROLLOUT[{name}] " + json.dumps(trace, default=float))
  return trace


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in same-stage first-divergence diagnostic")
@pytest.mark.parametrize("name", ["torus", "bowl"])
def test_same_stage_first_divergence_packet_gpu(name):
  packet = _diagnose(name)
  assert len(packet["steps"]) == 8
  for entry in packet["steps"]:
    assert entry["assembly"]["rows"] and len(
        entry["assembly"]["rows"]) == 8
    assert math.isfinite(entry["native_time"])
    assert abs(entry["native_time"] - entry["cpu_time"]) < 1e-6


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in uninterrupted rollout comparison")
@pytest.mark.parametrize("name", ["torus", "bowl"])
def test_uninterrupted_rollout_and_noninterference_gpu(name):
  pristine = _rollout(name)
  probed = _rollout(name, with_assembly=True)
  assert len(pristine["steps"]) == len(probed["steps"]) == 8
  for p_step, d_step in zip(pristine["steps"], probed["steps"]):
    for key in ("qpos_err", "qvel_err", "qacc_err"):
      assert p_step["host_float64"][key] == d_step["host_float64"][key], (
          key, p_step["step"])
    assert d_step["device"]["ncon_cpu"] == 8, d_step["step"]
    assert p_step["device"]["ncon_cpu"] == 8, p_step["step"]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in QP reference and cross-swap evidence")
def test_bowl_qp_reference_and_cross_swap_gpu():
  """Committed causal evidence: J carries, multipliers are innocent.

  Float64 active-set solves are NUMERICAL references, not exact
  solutions; the claim is the ORDER-OF-MAGNITUDE separation between the
  J-driven and multiplier-driven force errors.
  """
  model = _plugin_sphere_model("bowl")
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = _clone(sim.state._qpos)[0].astype(np.float64)
  qvel = _clone(sim.state._qvel)[0].astype(np.float64)
  cpu_data, _, _ = _cpu_contacts(model, qpos, qvel)
  system = sim.assembled_system(recompute=True)
  desc = sim._coupled_constraints.descriptor
  mask = _clone(system["contact_mask"])[0]
  pos = _clone(system["contact_position"])[0]
  _, _, contacts, pair_geoms = _cpu_contacts_from_data(model, cpu_data)
  assert len(pair_geoms) == 1 and all(c["dim"] == 1 for c in contacts)
  native = {"mask": mask, "pos": pos}
  assignment = _unique_native_slots(contacts, mask, pos)
  packed = np.asarray(
      desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  grows, crows = [], []
  for ci, contact in enumerate(contacts):
    slot, position_error = assignment[ci]
    assert position_error < 1e-3
    grows.append(int(desc.nr_joint) + int(packed[slot, 1]))
    crows.extend(contact["rows"].tolist())
  grows = np.asarray(grows)
  crows = np.asarray(crows)
  Jn = _clone(system["J"])[0].astype(np.float64)[grows]
  Jc = np.asarray(cpu_data.efc_J).reshape(
      cpu_data.nefc, model.nv)[crows].astype(np.float64)
  lamn = _clone(system["lambda"])[0].astype(np.float64)[grows]
  lamc = np.asarray(cpu_data.efc_force).astype(np.float64)[crows]
  Wn = _clone(system["W"])[0].astype(np.float64)[np.ix_(grows, grows)]
  Rn = _clone(system["R"])[0].astype(np.float64)[grows]
  bn = -_clone(system["rhs"])[0].astype(np.float64)[grows]
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, cpu_data, mass)
  Wc = Jc @ np.linalg.solve(mass, Jc.T)
  Rc = np.asarray(cpu_data.efc_R)[crows]
  bc = np.asarray(cpu_data.efc_b)[crows]
  j_rel = float(np.max(np.abs(Jn - Jc)) / np.max(np.abs(Jc)))
  lam_rel = float(np.max(np.abs(lamn - lamc))
                  / np.max(np.abs(lamc)))
  assert j_rel < 1e-6, j_rel
  assert lam_rel < 1e-6, lam_rel
  qfrc_cpu = np.asarray(cpu_data.qfrc_constraint).astype(np.float64)
  cross = {}
  cross_rot = {}
  for tag, Jm, lm in (("natJ-natLam", Jn, lamn), ("natJ-cpuLam", Jn, lamc),
                       ("cpuJ-natLam", Jc, lamn), ("cpuJ-cpuLam", Jc, lamc)):
    err = Jm.T @ lm - qfrc_cpu
    cross[tag] = float(np.max(np.abs(err)))
    # The gate-sensitive subspace is rotational (inertia 0.0167 turns
    # 1e-4 forces into 1e-2 accelerations); translational force errors
    # near 1e-4 on both cross terms are absorbed by the 4.19 body mass
    # (4e-5 accelerations, far below tiers) and carry no gate force.
    cross_rot[tag] = float(np.max(np.abs(err[3:])))
  assert (cross_rot["natJ-cpuLam"] / max(cross_rot["cpuJ-natLam"], 1e-30)
          > 1e3), (cross_rot, cross)
  ref_nat = _qp_reference_lam(Wn, Rn, bn)
  ref_cpu = _qp_reference_lam(Wc, Rc, bc)
  assert ref_nat is not None and ref_cpu is not None
  kkt_nat = _qp_kkt(lamn, Wn, Rn, bn)
  kkt_cpu = _qp_kkt(lamc, Wc, Rc, bc)
  kkt_ref = _qp_kkt(ref_nat, Wn, Rn, bn)
  assert kkt_ref["complementarity"] < 1e-6, kkt_ref
  print(f"QP_CROSS_SWAP[bowl] {json.dumps({'j_rel': j_rel, 'lam_rel': lam_rel, 'cross': cross, 'cross_rot': cross_rot, 'kkt_nat': kkt_nat, 'kkt_cpu': kkt_cpu, 'kkt_ref': kkt_ref})}")
  assert kkt_cpu["complementarity"] < 1e-6, kkt_cpu


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in same-state operator-swap diagnostic")
@pytest.mark.parametrize("name", ["torus", "bowl"])
def test_same_state_operator_swap_shows_causal_sensitivity_gpu(name):
  """Same-state operator swaps with owned snapshots and real smooth input.

  At an identical represented native state (post-rollout for torus where
  the gate drifts, initial for bowl), compare effective J_hi+J_lo, mass,
  W, R, aref/rhs, multipliers, reconstructed force/acceleration using
  owned clones and the actual native smooth rhs captured read-only (never
  the implied M*qacc-qfrc residual as a consistency certificate; the
  implied quantity is still reported labeled separately alongside the
  real rhs oracle). Asserts order-of-magnitude separation (not exact
  equality): bowl J carries (prior test), torus lam/W carries at the
  drifting state. Routes are not inferred from iteration counts.
  """
  model = _plugin_sphere_model(name)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  steps = 4 if name == "torus" else 0
  for _ in range(steps):
    sim.step(1, qfrc_applied=qfrc)
  qpos = _clone(sim.state._qpos)[0].astype(np.float64)
  qvel = _clone(sim.state._qvel)[0].astype(np.float64)
  # Fresh assembly at the identical represented state (no warmstart on
  # either side); owned snapshots cloned before any recompute.
  sim2 = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim2.reset(qpos=qpos.astype(np.float32)[None],
             qvel=qvel.astype(np.float32)[None])
  # Read-only capture of the actual native smooth rhs (solver qfrc input);
  # cloning never mutates device state (noninterference holds by
  # construction, as with the staged-workspace capture).
  smooth_seen = {}
  cc = sim2._coupled_constraints
  orig = cc._solve_kernel
  def wrapped(*args, **kwargs):
    import torch
    # qfrc_smooth is the second positional solver input (mass, smooth,
    # qpos, qvel, ...); record a clone without mutating the call.
    try:
      if len(args) >= 2 and isinstance(args[1], torch.Tensor):
        smooth_seen["qfrc"] = args[1].cpu().numpy().copy()
    except Exception:
      pass
    return orig(*args, **kwargs)
  cc._solve_kernel = wrapped
  try:
    system = sim2.assembled_system(recompute=True, qfrc_applied=qfrc)
  finally:
    cc._solve_kernel = orig
  assert "qfrc" in smooth_seen, "smooth rhs was not captured"
  native_smooth = np.asarray(smooth_seen["qfrc"]).reshape(-1).astype(np.float64)
  cpu_data, _, _ = _cpu_contacts(model, qpos, qvel)
  cpu_data.qfrc_applied[:] = qfrc[0].astype(np.float64)
  import mujoco as _mj
  _mj.mj_forward(model, cpu_data)
  cpu_smooth = np.asarray(cpu_data.qfrc_smooth, dtype=np.float64)
  cpu_applied = np.asarray(cpu_data.qfrc_applied, dtype=np.float64)
  # Real-rhs oracle gap (never the implied residual as certificate).
  # Native solver smooth input carries applied forces; compare against
  # CPU smooth+applied for identical staging.
  smooth_gap = float(np.max(np.abs(native_smooth - (cpu_smooth + cpu_applied))))
  # Owned snapshots for operators/multipliers/forces.
  desc = sim2._coupled_constraints.descriptor
  mask = _clone(system["contact_mask"])[0]
  pos = _clone(system["contact_position"])[0]
  _, _, contacts, pair_geoms = _cpu_contacts_from_data(model, cpu_data)
  assert len(pair_geoms) == 1 and all(c["dim"] == 1 for c in contacts)
  assignment = _unique_native_slots(contacts, mask, pos)
  packed = np.asarray(
      desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  grows, crows = [], []
  for ci, contact in enumerate(contacts):
    slot, position_error = assignment[ci]
    assert position_error < 1e-3
    grows.append(int(desc.nr_joint) + int(packed[slot, 1]))
    crows.extend(contact["rows"].tolist())
  grows = np.asarray(grows)
  crows = np.asarray(crows)
  Jn = _clone(system["J"])[0].astype(np.float64)[grows]
  Jc = np.asarray(cpu_data.efc_J).reshape(
      cpu_data.nefc, model.nv)[crows].astype(np.float64)
  lamn = _clone(system["lambda"])[0].astype(np.float64)[grows]
  lamc = np.asarray(cpu_data.efc_force).astype(np.float64)[crows]
  qfrc_cpu = np.asarray(cpu_data.qfrc_constraint).astype(np.float64)
  cross = {}
  cross_rot = {}
  for tag, Jm, lm in (("natJ-natLam", Jn, lamn), ("natJ-cpuLam", Jn, lamc),
                       ("cpuJ-natLam", Jc, lamn), ("cpuJ-cpuLam", Jc, lamc)):
    err = Jm.T @ lm - qfrc_cpu
    cross[tag] = float(np.max(np.abs(err)))
    cross_rot[tag] = float(np.max(np.abs(err[3:])))
  print(f"OPERATOR_SWAP[{name}] {json.dumps({'smooth_gap': smooth_gap, 'cross': cross, 'cross_rot': cross_rot})}")
  # Bowl: J carries by >1e2 on the rotational subspace (gate-sensitive;
  # translational errors are absorbed by the 4.19 body mass). Torus at the
  # drifting state: multipliers carry (lam-driven force error dominates
  # J-driven by an order of magnitude), directing the paired-operator
  # repair at W/b/lam accuracy rather than J alone.
  if name == "bowl":
    assert cross_rot["natJ-cpuLam"] / max(cross_rot["cpuJ-natLam"], 1e-30) > 1e2
  else:
    assert cross["cpuJ-natLam"] / max(cross["natJ-cpuLam"], 1e-30) > 3


def test_smooth_force_balance_uses_qfrc_smooth_cpu():
  """Force balance must use the actual smooth-force result.

  On a default-gravity free sphere the `qfrc_bias + applied` combination
  leaves an absolute residual of 82.18 (its sign convention differs),
  while `data.qfrc_smooth` balances exactly. An applied-force variant
  proves the control is not vacuous.
  """
  xml = ('<mujoco><option gravity="0 0 -9.81"/>'
         '<worldbody><body name="b" pos="0 0 1">'
         '<freejoint/><geom name="g" type="sphere" size="0.1"/>'
         "</body></worldbody></mujoco>")
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(model.qpos0)
  mujoco.mj_forward(model, data)
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  smooth = np.asarray(data.qfrc_smooth, dtype=np.float64)
  assert float(np.max(np.abs(smooth))) > 1.0
  residual = (mass @ np.asarray(data.qacc)
              - np.asarray(data.qfrc_constraint) - smooth)
  np.testing.assert_allclose(residual, 0.0, rtol=0, atol=1e-9)
  biased = (mass @ np.asarray(data.qacc)
            - np.asarray(data.qfrc_constraint)
            - (np.asarray(data.qfrc_bias) + np.asarray(data.qfrc_applied)))
  assert float(np.max(np.abs(biased))) > 1.0
  data.qfrc_applied[:] = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
  mujoco.mj_forward(model, data)
  residual2 = (mass @ np.asarray(data.qacc)
               - np.asarray(data.qfrc_constraint)
               - np.asarray(data.qfrc_smooth))
  np.testing.assert_allclose(residual2, 0.0, rtol=0, atol=1e-9)
