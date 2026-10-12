# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Transactional standalone sensor publication from generation-owned stages.

Host metadata selects addresses and admits storage. Sensor evaluation, history
reads, sleep/health masks and extension state all remain device operations.
History insertion and forward-stage evaluation belong to other public APIs.
"""
from dataclasses import fields, is_dataclass
import mujoco
import numpy as np


_FIELDS = ('_sensordata', '_raw_sensordata', '_energy', '_energy_computed',
           '_sensor_plugin_status')


def _admit(sim):
  """Charge resident backing, each rollback view and staged publication storage."""
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  from mujoco_metal.finite_difference import _plugin_snapshot_bytes
  from mujoco_metal.native_api import _owned_native_backing_bytes
  torch, seen = sim.state._torch, set()
  def count(value):
    if id(value) in seen:
      return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
      return value.numel()*value.element_size()
    if isinstance(value, dict):
      return sum(count(v) for v in value.values())
    if isinstance(value, (tuple, list)):
      return sum(count(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
      return sum(count(getattr(value, f.name)) for f in fields(value))
    if type(value).__module__.startswith('mujoco_metal.') and hasattr(value, '__dict__'):
      return count(vars(value))
    return 0
  # Distinct views may each be cloned even if they alias one resident storage.
  rollback = count(vars(sim))
  # External plugin classes need not belong to this package. Include their
  # actual tensor backings rather than assuming the package walker sees them.
  plugin_fields = [vars(plugin) for plugin in sim._native_plugins]
  rollback += sum(count(value) for value in plugin_fields)
  plugin_bytes = sum(_plugin_snapshot_bytes(p) for p in sim._native_plugins)
  b, n = sim.batch_size, int(sim._mjmodel.nsensordata)
  # Old/raw/staged/returned samples, per-address masks, lazy energy metadata,
  # plugin status and selection masks. Reserve a second copy of the complete
  # rollback graph for evaluator temporaries in this explicit API transaction.
  extra = 4*b*(12*n+32)
  resident = _owned_native_backing_bytes({'simulation': sim, 'plugins': plugin_fields})
  required = resident+2*rollback+3*plugin_bytes+extra
  budget = (sim.limits or CapacityLimits()).memory_budget_bytes
  if required > budget:
    raise CapacityOverflow(f'sensor stage publication needs {required} bytes; budget is {budget}')
  return required


def _evaluate_stage(sim, record, stage, mask):
  from mujoco_metal.forward_stages import ForwardStage
  pos = record.values[ForwardStage.POS]
  vel = record.values.get(ForwardStage.VEL)
  dynamics = pos['dynamics'] if vel is None else vel['dynamics']
  qvel = sim.state._qvel if vel is None else vel['qvel']
  solved = record.values.get(ForwardStage.CONSTRAINT)
  qacc = solved['qacc'] if stage == int(mujoco.mjtStage.mjSTAGE_ACC) else None
  sim._store_step_sensors(record.qpos, qvel, dynamics, qacc,
      stages=(stage,), compute_worlds=mask, position_context=pos['context'],
      include_spatial=False, finalize_energy=False)
  program = sim._raw_sensor_program
  poses = dict(dynamics['poses'], cvel=dynamics['cvel'], root_com=dynamics['root_com'])
  raw = sim._raw_sensordata
  if stage == int(mujoco.mjtStage.mjSTAGE_ACC) and sim._has_contact_sensors:
    raw = program.run_contact_device(poses, sim._contact_views(), out=raw, world_mask=mask)
  if stage == int(mujoco.mjtStage.mjSTAGE_POS):
    hull, info = program._sp_hull.reshape(-1), program._sp_hull_info.reshape(-1)
    if sim._has_ray_sensors:
      raw = program.run_rays_device(poses, hull, info, out=raw, world_mask=mask)
    if sim._has_geomdist_sensors:
      raw = program.run_geomdist_device(poses, hull, info, out=raw, world_mask=mask)
  sim._raw_sensordata = raw.clone()
  sim._sensordata = sim._history_program.samples(
      raw, sim.state._history, sim.state._time, kind=1)


def _publish(sim, values):
  """Binding-only commit boundary: previous borrowed diagnostic views survive."""
  for name, value in values.items():
    setattr(sim, name, value)


def publish_sensor_stage(sim, stage, record=None, env_ids=None):
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.native_api import _prepared_record, _selected_ids, _inverse_query_workspaces
  torch, model = sim.state._torch, sim._mjmodel
  stage = int(stage)
  minimum = {int(mujoco.mjtStage.mjSTAGE_POS): ForwardStage.POS,
             int(mujoco.mjtStage.mjSTAGE_VEL): ForwardStage.VEL,
             int(mujoco.mjtStage.mjSTAGE_ACC): ForwardStage.CONSTRAINT}[stage]
  record = _prepared_record(sim, minimum, record)
  # POS can run from a POS-only record. If a captured VEL bundle exists, the
  # evaluator consumes its motion fields and must validate those inputs too.
  if stage == int(mujoco.mjtStage.mjSTAGE_POS) and record.stage >= ForwardStage.VEL:
    record = _prepared_record(sim, ForwardStage.VEL, record)
  ids, _ = _selected_ids(sim, env_ids)
  _admit(sim)
  shape = (sim.batch_size, int(model.nsensordata))
  with torch.inference_mode(False), torch.no_grad():
    old = {name: getattr(sim, name) for name in _FIELDS}
    retained_values = {name: value.clone() if value is not None else None
                       for name, value in old.items()}
    previous = (old['_sensordata'].clone() if old['_sensordata'] is not None
                else torch.zeros(shape, dtype=torch.float32, device=sim.state._device))
    if sim._sensors is None or int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_SENSOR):
      return previous
    selected = torch.zeros((sim.batch_size,), dtype=torch.bool, device=sim.state._device)
    if ids is None:
      selected.fill_(True)
    else:
      selected[torch.as_tensor(ids, dtype=torch.long, device=selected.device)] = True
    selected &= sim.state._status == 0
    if record.row_valid is not None:
      selected &= record.row_valid
    row_status = record.values[minimum].get('status')
    if row_status is not None:
      selected &= row_status == 0
    addresses = np.zeros(model.nsensordata, bool)
    for index, need in enumerate(model.sensor_needstage):
      if int(need) == stage:
        first, width = int(model.sensor_adr[index]), int(model.sensor_dim[index])
        addresses[first:first+width] = True
    channels = torch.as_tensor(addresses, device=selected.device)
    with _inverse_query_workspaces(sim):
      adapter_before = [(plugin, plugin.device_snapshot()) for plugin in sim._native_plugins]
      _evaluate_stage(sim, record, stage, selected)
      accepted = selected & (sim._sensor_plugin_status == 0)
      writes = accepted[:, None] & channels[None, :]
      # One adapter's failure rejects this world's complete publication,
      # including otherwise-valid state changes in another stage adapter.
      for plugin, payload in adapter_before:
        plugin.restore_masked(payload, accepted)
      values = {}
      for name in _FIELDS:
        value = getattr(sim, name)
        if value is not None:
          if name in ('_sensordata', '_raw_sensordata'):
            retained = retained_values[name] if retained_values[name] is not None else previous
            value = torch.where(writes, value, retained)
          elif name in ('_energy', '_energy_computed'):
            retained = retained_values[name]
            if retained is None:
              retained = torch.zeros_like(value)
            value = torch.where(accepted[:, None], value, retained)
          values[name] = value.clone()
      plugin_states = [(p, p.device_snapshot()) for p in sim._native_plugins]
    plugin_before = [(p, p.device_snapshot()) for p in sim._native_plugins]
    try:
      _publish(sim, values)
      for plugin, payload in plugin_states:
        plugin.restore_device(payload)
      result = sim._sensordata.clone()
    except Exception:
      for name, value in old.items():
        setattr(sim, name, value)
      errors = []
      for plugin, payload in reversed(plugin_before):
        try:
          plugin.restore_device(payload)
        except Exception as error:
          errors.append(error)
      if errors:
        raise RuntimeError('sensor stage plugin rollback failed') from errors[0]
      raise
    return result
