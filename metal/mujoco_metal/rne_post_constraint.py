# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native batched counterpart of MuJoCo 3.10 ``mj_rnePostConstraint``.

The routine deliberately consumes a complete native forward constraint
result.  It reuses the RNE assembly and post kernels owned by
``SensorProgram``; no host MuJoCo physics is used by the query.

Pinned ``mj_rnePostConstraint`` source contributes applied body wrenches,
geom-backed contacts, and connect/weld equality wrenches to ``cfrc_ext``.
It explicitly skips flex contacts and advances over joint, tendon, and flex
equality rows without adding them to that external-wrench output. Actuator,
tendon, and other generalized forces affect the returned internal RNE fields
through the acceleration supplied by the constraint stage.
"""

from types import SimpleNamespace

import mujoco
import torch

_I32_MAX = (1 << 31) - 1


def _rne_allocation_bytes(model, batch_size):
  """Exact tensor bytes for a query-local RNE program and returned fields."""
  b = int(batch_size)
  nb, nv, nj, ng, ns, neq, nu = (
      int(model.nbody), int(model.nv), int(model.njnt), int(model.ngeom),
      int(model.nsite), int(model.neq), int(model.nu))
  elements = {
      "body_jnt": max(nb, 1) * 4,
      "dof_jntid": max(nv, 1),
      "joint_meta": max(nj, 1) * 2,
      "mass": max(nb, 1),
      "body_tree": max(nb, 1) * 2,
      "inertia": max(nb, 1) * 3,
      "body_weld": max(nb, 1),
      "geom_body": max(ng, 1),
      "site_body": max(ns, 1),
      "eq_meta": max(neq, 1) * 4,
      "eq_data": max(neq, 1) * 11,
      "site_pos": max(ns, 1) * 3,
      "actuator_transmission": max(nu, 1) * 2,
      "cacc_pair": 2 * b * max(nb, 1) * 6,
      "cfrc_int": b * max(nb, 1) * 6,
      "subtree_com": b * max(nb, 1) * 3,
      "cfrc_ext": b * max(nb, 1) * 6,
      "com_scratch": b * max(nb, 1) * 12,
      "qacc_low_zero": b * max(nv, 1),
      "small_metadata": 3 + 9 + 1,
      "result_spatial": b * max(nb, 1) * (6 + 6 + 6 + 3),
      "result_qacc": b * nv,
      "result_status": b,
      "dispatch_metadata": 32,
  }
  total = 0
  for name, count in elements.items():
    if count < 0 or count > _I32_MAX:
      raise ValueError(f"RNE post-constraint {name} exceeds int32 indexing")
    total += 4 * count
  # World-finiteness checks materialize one bool mask as large as a body
  # spatial field. Status combination and output masking retain a few
  # batch-sized masks plus two int32 status temporaries at peak.
  peak_bool = b * max(max(nb, 1) * 6, nv) + 6 * b
  peak_status = 2 * b
  if peak_bool + peak_status > _I32_MAX:
    raise ValueError("RNE post-constraint peak temporaries exceed int32 indexing")
  total += peak_bool + 4 * peak_status
  if total > _I32_MAX * 4:
    raise ValueError("RNE post-constraint workspace exceeds int32 byte indexing")
  return total


def _rne_result_bytes(model, batch_size):
  b, nb, nv = int(batch_size), int(model.nbody), int(model.nv)
  counts = (b * max(nb, 1) * 21, b * nv, b, 8)
  if any(value < 0 or value > _I32_MAX for value in counts):
    raise ValueError("RNE post-constraint result exceeds int32 indexing")
  peak_bool = b * max(max(nb, 1) * 6, nv) + 6 * b
  peak_status = 2 * b
  if peak_bool + peak_status > _I32_MAX:
    raise ValueError("RNE post-constraint peak temporaries exceed int32 indexing")
  total = 4 * sum(counts) + peak_bool + 4 * peak_status
  if total > _I32_MAX * 4:
    raise ValueError("RNE post-constraint result exceeds int32 byte indexing")
  return total


def _legacy_contact_workspace_bytes(sim):
  """Temporary RNE-layout packing needed by fixed-pair contact profiles."""
  contact = getattr(sim, "_contact", None)
  if contact is None:
    return 0
  b = int(sim.batch_size)
  n = int(contact.descriptor.pair_count)
  counts = (b * n * 11, n * 3, n * 5, 2 * n, n + 1)
  if any(value < 0 or value > _I32_MAX for value in counts):
    raise ValueError("legacy-contact RNE packing exceeds int32 indexing")
  total = 4 * sum(counts)
  if total > _I32_MAX * 4:
    raise ValueError("legacy-contact RNE packing exceeds int32 byte indexing")
  return total


def _query_memory_bytes(sim, model, batch_size, program):
  """Peak clone/workspace/output bytes admitted before query mutation."""
  from mujoco_metal.finite_difference import estimate_transaction_bytes
  total = estimate_transaction_bytes(sim)
  total += (_rne_result_bytes(model, batch_size) if program is not None
            else _rne_allocation_bytes(model, batch_size))
  total += _legacy_contact_workspace_bytes(sim)
  if total < 0 or total > _I32_MAX * 4:
    raise ValueError("RNE post-constraint peak allocation exceeds int32 indexing")
  return total


def _check_query_budget(required_bytes, budget_bytes):
  if required_bytes > budget_bytes:
    raise ValueError(
        f"RNE post-constraint query needs {required_bytes} bytes; "
        f"budget is {budget_bytes}")


def _validate_stage_result(sim, result):
  from mujoco_metal.forward_stages import ForwardStage

  if not isinstance(result, dict):
    raise TypeError("forward_result must be the bundle returned by mj_forward")
  record = result.get("record")
  record = sim.validate_forward_stage_record(record, ForwardStage.CONSTRAINT)
  constraint = record.values.get(ForwardStage.CONSTRAINT)
  position = record.values.get(ForwardStage.POS)
  velocity = record.values.get(ForwardStage.VEL)
  if (result.get("constraint") is not constraint
      or result.get("position") is not position
      or result.get("velocity") is not velocity
      or result.get("qacc") is not constraint.get("qacc")):
    raise ValueError("forward_result must reference this record's published stages")
  return record, position, velocity, constraint


def _make_rne_workspace(model, batch_size, device):
  """Build the minimal RNE buffers while reusing SensorProgram's constants."""
  from mujoco_metal.sensors import SensorProgram

  class Descriptor:
    nbody = int(model.nbody)
    njnt = int(model.njnt)
    nv = int(model.nv)
    ngeom = int(model.ngeom)
    nsite = int(model.nsite)
    ntendon = int(model.ntendon)
    nu = int(model.nu)
    sensor_type = ()

  class Workspace:
    pass

  workspace = Workspace()
  workspace.descriptor = Descriptor()
  workspace._torch = torch
  workspace._device = torch.device(device)
  workspace.batch_size = int(batch_size)
  workspace._s_dummy = torch.zeros(1, dtype=torch.float32,
                                   device=workspace._device)
  # This shared builder owns only compiled model metadata and RNE device
  # buffers. It does not lower or execute model sensors.
  SensorProgram._build_rne_constants(workspace, model)
  return workspace


def _validate_mps(value, shape, name, device):
  if (not isinstance(value, torch.Tensor) or value.device.type != "mps"
      or (device.index is not None and value.device.index != device.index)
      or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape)
      or not value.is_contiguous()):
    raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
  return value


def _velocity_cvel(dynamics):
  """Return the VEL product stored beside ``poses`` by SmoothDynamics."""
  if not isinstance(dynamics, dict) or dynamics.get("cvel") is None:
    raise ValueError("velocity dynamics are missing cvel")
  return dynamics["cvel"]


def _run_rne_post(program, sim, qvel, qacc, dynamics, qacc_low,
                  legacy_contact_result=None):
  """Run existing force-assembly and RNE-post pipelines; return borrowed data."""
  model, torch = sim.model, program._torch
  b, nb, nv = int(sim.batch_size), int(model.nbody), int(model.nv)
  dev = program._device
  _validate_mps(qvel, (b, nv), "qvel", dev)
  _validate_mps(qacc, (b, nv), "qacc", dev)
  if nv == 0 or qacc_low is None:
    qacc_low = program._rne_qacc_low_zero
  _validate_mps(qacc_low, (b, max(nv, 1)), "qacc_low", dev)
  poses = dynamics.get("poses")
  if not isinstance(poses, dict):
    raise ValueError("velocity stage is missing poses")
  for name, shape in (
      ("body_pos", (b, nb, 3)), ("body_quat", (b, nb, 4)),
      ("inertial_pos", (b, nb, 3)), ("inertial_quat", (b, nb, 4)),
      ("joint_anchor", (b, int(model.njnt), 3)),
      ("joint_axis", (b, int(model.njnt), 3))):
    _validate_mps(poses.get(name), shape, f"poses[{name!r}]", dev)
  # SmoothDynamics publishes cvel beside its poses dictionary. Keeping this
  # lookup at the actual VEL-stage boundary matters for every nonzero-body
  # model: cvel is not part of the nested pose bundle.
  cvel = _velocity_cvel(dynamics)
  _validate_mps(cvel, (b, nb, 6), "dynamics['cvel']", dev)

  cc = getattr(sim, "_coupled_constraints", None)
  lam_raw = eq_rowadr = None
  lam_nr = lam_stride = 0
  if cc is not None:
    lam_nr = int(cc.descriptor.nr)
    lam_stride = int(cc._debug_stride)
    lam_raw = cc._workspace["workspace_debug"].reshape(-1)
    eq_rowadr = cc._constants["eq_rowadr"].reshape(-1)
  contact = sim._contact_views()
  if contact is None and getattr(sim, "_contact", None) is not None:
    contact = _legacy_contact_views(program, sim, legacy_contact_result)
  dummy = program._s_dummy
  if contact is None:
    cfr = cfo = crow = cmu = dummy
    cpk = torch.zeros(3, dtype=torch.int32, device=dev)
    pge = torch.zeros(2, dtype=torch.int32, device=dev)
    pof = torch.zeros(2, dtype=torch.int32, device=dev)
    nc = npairs = 0
  else:
    nc = int(contact["frame"].shape[1])
    npairs = int(contact.get("npairs", 0))
    cfr = contact["frame"].reshape(-1)
    cfo = contact["force"].reshape(-1)
    crow = contact["row"].contiguous().reshape(-1)
    cpk = contact["packed"].reshape(-1)
    cmu = contact["mu"].reshape(-1)
    pge = contact["pair_geoms"].reshape(-1)
    pof = contact["pair_offset"].reshape(-1)
  xfrc = getattr(sim, "_body_wrench", None)
  XF = xfrc.reshape(-1) if xfrc is not None else dummy
  has_x = int(xfrc is not None)
  eqr = (eq_rowadr if eq_rowadr is not None else
         torch.zeros(1, dtype=torch.int32, device=dev))
  adims = torch.tensor([b, nb, int(eqr.numel()) if eq_rowadr is not None else 0,
                        nc, npairs, has_x, lam_nr, lam_stride],
                       dtype=torch.int32, device=dev)
  bjn = program._rne_body_jnt[:, 1].contiguous().reshape(-1)
  program._rne_assemble(
      XF, cfr, cfo, crow, cpk, cmu, pge, pof,
      program._rne_geom_bodyid, bjn,
      program._rne_eq_meta.reshape(-1), eqr,
      program._rne_eq_data.reshape(-1), program._rne_site_lpos.reshape(-1),
      program._rne_site_bodyid,
      lam_raw.reshape(-1) if lam_raw is not None else dummy,
      poses["body_pos"].reshape(-1), poses["body_quat"].reshape(-1),
      poses["inertial_pos"].reshape(-1), program._rne_mass,
      program._rne_body_tree.reshape(-1), program._rne_ext.reshape(-1),
      adims, program._rne_com_scratch.reshape(-1),
      threads=(b,), group_size=(1,))

  gravity_disabled = bool(int(model.opt.disableflags)
      & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
  rdims = torch.tensor([b, nb, nv, int(model.njnt), int(gravity_disabled)],
                       dtype=torch.int32, device=dev)
  program._rne_post(
      qvel.reshape(-1) if nv else dummy,
      qacc.reshape(-1) if nv else dummy,
      cvel.reshape(-1), program._rne_mass,
      program._rne_inertia.reshape(-1), program._rne_body_tree.reshape(-1),
      program._rne_body_jnt.reshape(-1),
      program._rne_jnt_type if model.njnt else torch.zeros(
          1, dtype=torch.int32, device=dev),
      program._rne_jnt_dofadr if model.njnt else torch.zeros(
          1, dtype=torch.int32, device=dev),
      poses["joint_anchor"].reshape(-1) if model.njnt else dummy,
      poses["joint_axis"].reshape(-1) if model.njnt else dummy,
      poses["inertial_pos"].reshape(-1), poses["inertial_quat"].reshape(-1),
      poses["body_quat"].reshape(-1), program._rne_ext.reshape(-1),
      program._rne_cacc_pair.reshape(-1), program._rne_cfrc.reshape(-1),
      program._rne_scom.reshape(-1), rdims, program._rne_gravity,
      program._rne_com_scratch.reshape(-1),
      qacc_low.reshape(-1) if nv else dummy,
      threads=(b,), group_size=(1,))
  return {"cacc": program._rne_cacc,
          "cfrc_ext": program._rne_ext,
          "cfrc_int": program._rne_cfrc,
          "subtree_com": program._rne_scom}


def _legacy_contact_views(program, sim, result):
  """Pack an admitted legacy fixed-pair contact result for the shared RNE MSL.

  The legacy profile has at most five contact rows per pair. Its solved force
  rows already use the same normal/pyramidal facet positions consumed by the
  shared contact-to-wrench shader, but have a five-float rather than eleven-
  float slot stride. Pack that prefix and the fixed model metadata without
  routing physics through the host.
  """
  contact = sim._contact
  workspace = contact._workspace
  d = contact.descriptor
  torch, b, n = program._torch, int(sim.batch_size), int(d.pair_count)
  if result is None:
    raise ValueError("legacy contact post query requires the published contact stage")
  mask = result.get("mask")
  if (not isinstance(mask, torch.Tensor) or mask.device.type != "mps"
      or mask.dtype != torch.float32 or tuple(mask.shape) != (b, n)
      or not mask.is_contiguous()):
    raise ValueError("legacy contact stage has invalid contact mask")
  raw_force = workspace["force"][:b * n * 5].reshape(b, n, 5)
  force = torch.zeros((b, n, 11), dtype=torch.float32, device=program._device)
  force[..., :5].copy_(raw_force)
  packed = torch.zeros((n, 3), dtype=torch.int32, device=program._device)
  if n:
    packed[:, 0].copy_(contact._constants["condim"])
    packed[:, 2].fill_(int(mujoco.mjtCone.mjCONE_PYRAMIDAL))
  mu = torch.zeros((n, 5), dtype=torch.float32, device=program._device)
  if n:
    mu[:, :2].copy_(contact._constants["friction"])
  pair_geoms = torch.stack((contact._constants["geom1"],
                            contact._constants["geom2"]), dim=1).contiguous()
  pair_offset = torch.arange(n + 1, dtype=torch.int32, device=program._device)
  return {"frame": workspace["frame"][:b * n * 12].reshape(b, n, 12),
          "force": force, "row": mask, "packed": packed.reshape(-1),
          "mu": mu.reshape(-1), "pair_geoms": pair_geoms.reshape(-1),
          "pair_offset": pair_offset, "npairs": n}


def _finish_rne_outputs(raw, qvel, qacc, status):
  """Create independent outputs and isolate failed rows, including empty B."""
  b = int(qvel.shape[0])
  # MPS can report an empty-axis ``all`` as false even though the IEEE
  # reduction identity is true. A zero-DOF model has [B,0] qvel/qacc, so
  # initialize the per-world result to true and only reduce tensors with at
  # least one value. ``numel`` is shape metadata and does not synchronize.
  finite = torch.ones((b,), dtype=torch.bool, device=qvel.device)
  for value in (qvel, qacc, *raw.values()):
    if value.numel():
      finite = finite & torch.isfinite(value).flatten(1).all(dim=1)
  status = torch.where((status == 0) & ~finite,
                       torch.full_like(status, 3), status)
  failed_rows = status != 0
  failed_spatial = failed_rows.reshape(b, 1, 1)
  output = {}
  for name, value in raw.items():
    output[name] = value.clone()
    output[name].masked_fill_(failed_spatial, 0.0)
  output["qacc"] = qacc.clone()
  output["qacc"].masked_fill_(failed_rows[:, None], 0.0)
  output["status"] = status.clone()
  return output


def mj_rnePostConstraint(sim, *, forward_result=None, qacc=None):
  """Return owned MPS post-constraint RNE fields for the current state.

  A fresh, full native forward constraint pipeline is evaluated when
  ``forward_result`` is omitted. A caller may instead supply the coherent
  bundle returned by ``native_api.mj_forward``. ``qacc`` optionally replaces
  that bundle's acceleration, matching the source routine's use of the
  caller-owned ``mjData.qacc`` field. Returned fields use MuJoCo's rot:lin
  spatial convention and have leading batch dimension.

  The query restores simulation and plugin workspaces on success or failure.
  It performs no host numerical physics and does not mutate persistent state.
  """
  if not hasattr(sim, "_state") or not isinstance(sim.model, mujoco.MjModel):
    raise TypeError("sim must be a MetalSimulation")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("mj_rnePostConstraint requires pinned MuJoCo 3.10.0")
  from mujoco_metal.native_api import _inverse_query_workspaces

  b, model, state = int(sim.batch_size), sim.model, sim.state
  nv = int(model.nv)
  if qacc is not None:
    if (not isinstance(qacc, torch.Tensor) or qacc.device.type != "mps"
        or qacc.dtype != torch.float32 or tuple(qacc.shape) != (b, nv)
        or not qacc.is_contiguous()):
      raise ValueError(f"qacc must be contiguous float32 MPS with shape {(b, nv)}")

  # Admit the reusable transaction snapshot, lazy no-sensor workspace, and
  # caller-owned outputs before shader compilation or query mutation.
  program = getattr(sim, "_sensors", None)
  query_bytes = _query_memory_bytes(sim, model, b, program)
  budget = int(getattr(getattr(sim, "limits", None),
                       "memory_budget_bytes", 1 << 30))
  _check_query_budget(query_bytes, budget)

  with _inverse_query_workspaces(sim):
    # Existing sensor execution already owns the exact RNE kernels and
    # buffers. The sensor-free route allocates its bounded workspace only
    # after the transaction has admitted plugin rollback and preserved the
    # complete simulation graph.
    if program is None:
      program = _make_rne_workspace(model, b, state._device)
    if forward_result is None:
      from mujoco import mjtStage
      forward_result = sim.forward_skip(mjtStage.mjSTAGE_NONE, True)
    record, position, velocity, constraint = _validate_stage_result(
        sim, forward_result)
    dynamics = velocity["dynamics"]
    effective_qacc = constraint["qacc"] if qacc is None else qacc
    if qacc is not None:
      qacc_low = None
    else:
      qacc_low = constraint.get("qacc_low")
      if nv == 0:
        qacc_low = None
    raw = _run_rne_post(program, sim, velocity["qvel"], effective_qacc,
                        dynamics, qacc_low,
                        constraint.get("contact_result"))
    output = _finish_rne_outputs(
        raw, velocity["qvel"], effective_qacc, constraint["status"])
  return output
