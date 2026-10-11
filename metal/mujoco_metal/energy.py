# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native energy-stage storage and transactional prepared-stage publication."""
from types import SimpleNamespace
import numpy as np
import mujoco


def energy_storage_enabled(model):
    """Compiled allocation policy shared by host admission and construction."""
    return bool(int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_ENERGY)
                or np.any(np.isin(np.asarray(getattr(model, 'sensor_type', [])),
                    [int(mujoco.mjtSensor.mjSENS_E_POTENTIAL),
                     int(mujoco.mjtSensor.mjSENS_E_KINETIC)])))


def energy_storage_bytes(model, batch_size):
    """Exact metadata/storage allocation size; no state values inspected."""
    edges = 0
    for fid in range(int(model.nflex)):
        if bool(model.flex_rigid[fid]) or int(model.flex_dim[fid]) > 1:
            continue
        if float(model.flex_edgestiffness[fid]) == 0.0:
            continue
        begin = int(model.flex_edgeadr[fid])
        edges += sum(not bool(model.flexedge_rigid[edge])
                     for edge in range(begin, begin + int(model.flex_edgenum[fid])))
    # Two output/staging columns; masses, gravity, spring reference;
    # joint/tendon metadata, flex rows, and 12 + batch dispatch dimensions.
    return (16 * batch_size + 4 * max(int(model.nbody) - 1, 1) + 12
            + 4 * max(int(model.nq), 1) + 24 * max(int(model.njnt), 1)
            + 36 * max(int(model.ntendon), 1) + 12 * max(edges, 1)
            + 4 * (12 + batch_size))


def _build_energy_storage(sim):
    torch = sim.state._torch
    model, batch_size = sim._mjmodel, sim.batch_size
    from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
    from mujoco_metal.native_api import _owned_native_backing_bytes
    limit = (getattr(sim, "limits", None) or CapacityLimits()).memory_budget_bytes
    if _owned_native_backing_bytes(sim) + energy_storage_bytes(model, batch_size) > limit:
        raise CapacityOverflow("energy stage storage exceeds owned memory budget")
    stage = SimpleNamespace(_state=sim.state, _spatial_tendons=sim._spatial_tendons)
    stage._energy = torch.zeros((batch_size, 2), dtype=torch.float32,
                                device=sim.state._device)
    stage._energy_stage = torch.empty_like(stage._energy)
    stage._energy_body_mass = None
    stage._energy_gravity = None
    stage._energy_qpos_spring = None
    stage._energy_body_mass = torch.zeros(
        (max(int(model.nbody) - 1, 1),), dtype=torch.float32,
        device=stage._state._device)
    if model.nbody > 1:
      stage._energy_body_mass[:model.nbody - 1].copy_(torch.as_tensor(
          np.asarray(model.body_mass[1:], dtype=np.float32).copy(),
          dtype=torch.float32, device=stage._state._device))
    stage._energy_gravity = torch.as_tensor(
        np.asarray(model.opt.gravity, dtype=np.float32).copy(),
        dtype=torch.float32, device=stage._state._device)
    stage._energy_qpos_spring = torch.zeros(
        (max(int(model.nq), 1),), dtype=torch.float32,
        device=stage._state._device)
    if model.nq:
      stage._energy_qpos_spring[:model.nq].copy_(torch.as_tensor(
          np.asarray(model.qpos_spring, dtype=np.float32).copy(),
          dtype=torch.float32, device=stage._state._device))
    energy_jint = np.zeros((max(int(model.njnt), 1), 3), dtype=np.int32)
    energy_jfloat = np.zeros((max(int(model.njnt), 1), 3), dtype=np.float32)
    for jid in range(int(model.njnt)):
      energy_jint[jid] = (int(model.jnt_type[jid]),
                          int(model.jnt_qposadr[jid]),
                          int(model.jnt_bodyid[jid]))
      energy_jfloat[jid] = (
          float(model.jnt_stiffness[jid]),
          float(np.asarray(model.jnt_stiffnesspoly).reshape(-1, 2)[jid, 0]),
          float(np.asarray(model.jnt_stiffnesspoly).reshape(-1, 2)[jid, 1]))
    energy_tint = np.zeros((max(int(model.ntendon), 1), 4), dtype=np.int32)
    energy_tfloat = np.zeros((max(int(model.ntendon), 1), 5), dtype=np.float32)
    tendon_poly = np.asarray(model.tendon_stiffnesspoly).reshape(
        int(model.ntendon), 2) if int(model.ntendon) else np.zeros((0, 2))
    tendon_range = np.asarray(model.tendon_lengthspring).reshape(
        int(model.ntendon), 2) if int(model.ntendon) else np.zeros((0, 2))
    paths = (stage._spatial_tendons._meta.paths
             if stage._spatial_tendons is not None else [None] * int(model.ntendon))
    for tid in range(int(model.ntendon)):
      tree_num = int(np.asarray(model.tendon_treenum)[tid])
      treeids = np.asarray(model.tendon_treeid).reshape(
          int(model.ntendon), 2)[tid]
      energy_tint[tid] = (tree_num, int(treeids[0]), int(treeids[1]),
                          int(paths[tid] is not None))
      energy_tfloat[tid] = (
          float(model.tendon_stiffness[tid]),
          float(tendon_poly[tid, 0]), float(tendon_poly[tid, 1]),
          float(tendon_range[tid, 0]), float(tendon_range[tid, 1]))
    edge_pairs = []
    flex_dim = np.asarray(model.flex_dim)
    flex_rigid = np.asarray(model.flex_rigid)
    edge_adr = np.asarray(model.flex_edgeadr)
    edge_num = np.asarray(model.flex_edgenum)
    edge_rigid = np.asarray(model.flexedge_rigid)
    rest = np.asarray(model.flexedge_length0, dtype=np.float32)
    edge_stiffness = np.asarray(model.flex_edgestiffness, dtype=np.float32)
    for fid in range(int(model.nflex)):
      if bool(flex_rigid[fid]) or int(flex_dim[fid]) > 1:
        continue
      for edge in range(int(edge_adr[fid]),
                        int(edge_adr[fid]) + int(edge_num[fid])):
        if not bool(edge_rigid[edge]) and float(edge_stiffness[fid]) != 0.0:
          edge_pairs.append((edge, float(rest[edge]),
                             float(edge_stiffness[fid])))
    stage._energy_joint_int = torch.as_tensor(
        energy_jint.reshape(-1), dtype=torch.int32, device=stage._state._device)
    stage._energy_joint_float = torch.as_tensor(
        energy_jfloat.reshape(-1), dtype=torch.float32, device=stage._state._device)
    stage._energy_tendon_int = torch.as_tensor(
        energy_tint.reshape(-1), dtype=torch.int32, device=stage._state._device)
    stage._energy_tendon_float = torch.as_tensor(
        energy_tfloat.reshape(-1), dtype=torch.float32, device=stage._state._device)
    flex_int = np.zeros((max(len(edge_pairs), 1),), dtype=np.int32)
    flex_float = np.zeros((max(len(edge_pairs), 1), 2), dtype=np.float32)
    stage._energy_flex_row_count = len(edge_pairs)
    if edge_pairs:
      flex_int[:len(edge_pairs)] = [row[0] for row in edge_pairs]
      flex_float[:len(edge_pairs)] = [row[1:] for row in edge_pairs]
    stage._energy_flex_int = torch.as_tensor(
        flex_int, dtype=torch.int32, device=stage._state._device)
    stage._energy_flex_float = torch.as_tensor(
        flex_float.reshape(-1), dtype=torch.float32,
        device=stage._state._device)
    stage._energy_position_dims = torch.tensor(
        [int(model.nq), int(model.nv), int(model.nbody), int(model.njnt),
         int(model.ntendon), int(model.nflexedge), len(edge_pairs),
         max(int(model.ntree), 1),
         int(bool(int(model.opt.disableflags)
                  & int(mujoco.mjtDisableBit.mjDSBL_SPRING))),
         int(bool(int(model.opt.disableflags)
                  & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))),
         int(bool(int(model.opt.enableflags)
                  & int(mujoco.mjtEnableBit.mjENBL_SLEEP))),
         int(batch_size)] + [1] * int(batch_size),
        dtype=torch.int32, device=stage._state._device)
    return {key: value for key, value in vars(stage).items()
            if key.startswith("_energy")}


def build_energy_storage(sim):
    # Persistent buffers must remain writable after an inference-mode caller.
    torch = sim.state._torch
    with torch.inference_mode(False), torch.no_grad():
        return _build_energy_storage(sim)


def sample_sensor_energy(sim, program, output, qpos, qvel, dynamics,
                         context, stages, world_mask, *, history_advance=False):
    """Sample canonical native energies before sensor cutoff/history.

    Both energy sensors belong to the pinned POS stage. Kinetic energy uses
    the live velocity and the already prepared mass, even before VEL refresh.
    Sensor tick masks govern data.energy as well as the returned samples.
    """
    if int(mujoco.mjtStage.mjSTAGE_POS) not in tuple(int(s) for s in stages):
        return
    torch, model = sim.state._torch, sim._mjmodel
    potential = torch.zeros(sim.batch_size, dtype=torch.bool,
                            device=sim.state._qpos.device)
    active = sim.state._status == 0
    if world_mask is not None:
        active = active & world_mask
    compute = sim._history_program.sensor_compute_mask(
        sim.state._history, sim.state._time)
    awake = getattr(program, '_sensor_awake_mask', None)
    for index, raw_type in enumerate(program.descriptor.sensor_type):
        typ = int(raw_type)
        if typ not in (int(mujoco.mjtSensor.mjSENS_E_POTENTIAL),
                       int(mujoco.mjtSensor.mjSENS_E_KINETIC)):
            continue
        adr = int(program.descriptor.sensor_adr[index])
        if history_advance and not (int(model.sensor_history[index, 0]) > 0
                                    and float(model.sensor_delay[index]) > 0):
            continue
        mask = active & compute[:, adr]
        if awake is not None:
            mask = mask & (awake[:, index] != 0)
        component = int(typ == int(mujoco.mjtSensor.mjSENS_E_KINETIC))
        already_computed = (history_advance and int(model.opt.enableflags)
                            & int(mujoco.mjtEnableBit.mjENBL_ENERGY))
        if component == 0:
            potential |= mask
            if not already_computed:
                sim._update_energy_position(qpos, dynamics, context,
                                            world_mask=mask.contiguous(), manual=True)
        elif not already_computed:
            sim._update_energy_velocity({'dynamics': dynamics, 'qvel': qvel},
                                        world_mask=mask.contiguous(), manual=True)
        value = sim._energy[:, component]
        cutoff = float(program.descriptor.sensor_cutoff[index])
        if cutoff > 0:
            value = value.clamp(-cutoff, cutoff)
        output[:, adr] = torch.where(mask, value, output[:, adr])
    if (not history_advance and not int(model.opt.enableflags)
        & int(mujoco.mjtEnableBit.mjENBL_ENERGY)):
        # engine_forward clears both values after sensorPos when no potential
        # energy sensor actually computed this sweep (including kinetic-only).
        sim._energy[active & ~potential] = 0.0


def advance_rk_energy_history(sim, dynamics, qpos, qvel, world_mask):
    """Pinned mj_advance energy sampling with X3 geometry and restored X0."""
    if sim._energy is None or sim._raw_sensordata is None:
        return
    sample_sensor_energy(sim, sim._raw_sensor_program, sim._raw_sensordata,
                         qpos, qvel, dynamics, None,
                         (mujoco.mjtStage.mjSTAGE_POS,), world_mask,
                         history_advance=True)


def _publish_energy(sim, target, working):
    """Commit only the two diagnostic columns; fault-injection boundary."""
    # Keep earlier borrowed forward-stage energy views unchanged.
    sim._energy = working
    sim._energy_manual = True


def update_energy(sim, component, record=None, *, env_ids=None):
    """Publish one energy column from a coherent prepared POS/VEL record.

    Explicit publication is independent of the automatic ENERGY enable bit.
    Temporary mass-product and mask buffers are restored before returning, so
    the supplied forward record and earlier borrowed outputs remain usable.
    """
    from mujoco_metal.forward_stages import ForwardStage
    from mujoco_metal.native_api import _prepared_record, _selected_ids, _owned_native_backing_bytes
    from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
    torch = sim.state._torch
    minimum = ForwardStage.POS if component == 0 else ForwardStage.VEL
    record = _prepared_record(sim, minimum, record)
    ids, _ = _selected_ids(sim, env_ids)
    position = record.values[ForwardStage.POS]
    velocity = record.values.get(ForwardStage.VEL)
    model, batch = sim._mjmodel, sim.batch_size
    old_fields = {key: value for key, value in vars(sim).items() if key.startswith('_energy')}
    old_energy = getattr(sim, '_energy', None)
    if old_energy is not None and (torch.is_inference(old_energy) or
        tuple(old_energy.shape) != (batch, 2) or old_energy.dtype != torch.float32 or
        old_energy.device != sim.state._qpos.device or not old_energy.is_contiguous()):
        raise ValueError('energy requires mutable contiguous float32 device storage')
    workspace = sim._smooth._workspace
    scratch = []
    seen = set()
    def own(value):
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            if torch.is_inference(value):
                raise ValueError('energy scratch requires mutable device storage')
            seen.add(id(value))
            scratch.append(value)
    for key in ('energy_velocity_dims', 'mass_matvec_dims', 'mass_sparse_product'):
        own(workspace.get(key))
    own(getattr(sim.state, '_reset_mask_i32', None))
    for key in ('_energy_stage', '_energy_position_dims'):
        own(getattr(sim, key, None))
    # Account for rollback clones, staged/returned energy, selection mask/IDs
    # and the source body/tree awake lists used by potential energy publication.
    extra = sum(value.numel() * value.element_size() for value in scratch)
    extra += 4 * batch * 5 + 8 * batch + 8 * batch
    if old_energy is None:
        extra += energy_storage_bytes(model, batch)
    limits = getattr(sim, 'limits', None) or CapacityLimits()
    if _owned_native_backing_bytes(sim) + extra > limits.memory_budget_bytes:
        raise CapacityOverflow('energy publication exceeds owned memory budget')
    new_fields = build_energy_storage(sim) if old_energy is None else None
    target = new_fields['_energy'] if new_fields is not None else old_energy
    with torch.inference_mode(False), torch.no_grad():
        old_values = target.clone()
        working = target.clone()
        saved = [(value, value.clone()) for value in scratch]
    old_version = target._version
    mask = torch.zeros((batch,), dtype=torch.bool, device=sim.state._qpos.device)
    if ids is None:
        mask.fill_(True)
    else:
        mask[torch.as_tensor(ids, dtype=torch.long, device=mask.device)] = True
    if record.row_valid is not None:
        mask &= record.row_valid
    mask &= sim.state._status == 0
    values = position if component == 0 else velocity
    status = values.get('status', values['dynamics'].get('status'))
    if status is not None:
        mask &= status == 0
    committed = False
    try:
        if new_fields is not None:
            vars(sim).update(new_fields)
        sim._energy = working
        if component == 0:
            sim._update_energy_position(record.qpos, position['dynamics'],
                                        position.get('context'), world_mask=mask, manual=True)
        else:
            sim._update_energy_velocity(velocity, world_mask=mask, manual=True)
        with torch.no_grad():
            _publish_energy(sim, target, working)
        result = working[:, component].clone()
        committed = True
    except Exception:
        if target._version != old_version:
            target.copy_(old_values)
        for name in tuple(vars(sim)):
            if name.startswith('_energy') and name not in old_fields:
                delattr(sim, name)
        vars(sim).update(old_fields)
        raise
    finally:
        if committed:
            sim._energy = working
        for value, saved_value in saved:
            value.copy_(saved_value)
    return result
