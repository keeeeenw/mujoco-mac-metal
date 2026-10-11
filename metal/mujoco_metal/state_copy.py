# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Selected batched state copying without forward evaluation.

Selected state copying follows engine_support.c's plain component copy. A
setter's admission checks must not normalize or reject source state values.
"""
import numpy as np
import torch

from mujoco_metal.native_api import StateSpec, _STATE_FIELDS, _selected_ids


def _backing(sim, attr):
  owner = sim if attr in ('_control', '_applied_force', '_body_wrench') else sim.state
  return getattr(owner, attr, None)


def _shape(model, attr, batch):
  tails = {
      '_time': (), '_qpos': (int(model.nq),), '_qvel': (int(model.nv),),
      '_act': (int(model.na),), '_history': (int(model.nhistory),),
      '_qacc_warmstart': (int(model.nv),), '_control': (int(model.nu),),
      '_applied_force': (int(model.nv),), '_body_wrench': (int(model.nbody), 6),
      '_eq_active': (int(model.neq),), '_mpos': (int(model.nmocap), 3),
      '_mquat': (int(model.nmocap), 4), '_userdata': (int(model.nuserdata),),
      '_plugin_state': (int(model.npluginstate),),
  }
  return (batch, *tails[attr])


def _effective_precision(precision, high, position):
  """Reconcile a private source component without changing its owner/witness."""
  if position:
    span = slice(0, 2 * precision.nq)
    shadow, original = precision.qpos_shadow, precision._qpos
  else:
    span = slice(2 * precision.nq, None)
    shadow, original = precision.qvel_shadow, precision._qvel
  words = precision.words[:, span]
  changed = (high.view(torch.int32) != shadow.view(torch.int32)).any(dim=1)
  if high is not original:
    changed.fill_(True)
  return torch.where(changed[:, None], 0., words), span


def _copy_metadata(sim):
  names = ('_assembled_system_valid', '_accepted_step', '_last_coupled',
           '_last_coupled_generation', '_spatial_cache_key', '_spatial_kin',
           '_step1_record')
  values = {name: getattr(sim, name) for name in names if hasattr(sim, name)}
  stages = getattr(sim, '_forward_stages', None)
  stage_values = None if stages is None else dict(vars(stages))
  record = None if stages is None else stages._record
  coherent = []
  if record is not None:
    for name, value in record.input_tensors.items():
      expected = record.input_versions.get(name)
      if expected is not None and stages._version(value) == expected:
        coherent.append((name, value))
  caches = []
  for name in ('_smooth', '_coupled_constraints'):
    owner = getattr(sim, name, None)
    attrs = {key: getattr(owner, key) for key in ('_position_context_epoch', '_position_context_valid')
             if owner is not None and hasattr(owner, key)}
    if attrs:
      caches.append((owner, attrs))
  return values, stages, stage_values, record, coherent, sim.state._generation, caches


def _invalidate(sim):
  # A selected copy never evaluates FK, flex, tendon, mass, contacts or solve.
  # Those outputs remain physically untouched and are recomputed on demand.
  sim._invalidate_forward_stage_record()
  sim.state._generation += 1
  sim._assembled_system_valid = False
  sim._accepted_step = None
  sim._last_coupled = None
  sim._last_coupled_generation = None
  sim._spatial_cache_key = None
  sim._spatial_kin = None
  for name in ('_smooth', '_coupled_constraints'):
    owner = getattr(sim, name, None)
    if owner is not None and hasattr(owner, '_position_context_epoch'):
      owner._position_context_epoch += 1
    if owner is not None and hasattr(owner, '_position_context_valid'):
      owner._position_context_valid = False


def copy_state(src, dst, spec=StateSpec.ALL, *, src_env_ids=None, dst_env_ids=None):
  """Copy paired world components between identical compiled model owners.

  All inputs are device tensors already owned by their simulations. State bits
  use pinned mjtState ordering. Explicit selections pair in supplied order;
  overlapping same-simulation selections read all source rows before writing.
  QPOS/QVEL include their retained words. No unrelated physical state, plugin
  callbacks, forward producers or history cursors are evaluated. Zero-width
  selections and signature zero have no effects. Success invalidates borrowed
  forward records; failure restores coherent records as well as copied data.
  """
  if (isinstance(spec, (bool, np.bool_)) or
      not isinstance(spec, (int, np.integer)) or int(spec) < 0 or
      int(spec) & ~StateSpec.ALL):
    raise ValueError('spec contains bits outside pinned mjtState')
  spec = int(spec)
  sm, dm = src._mjmodel, dst._mjmodel
  if (not src.state._model_fingerprint or
      src.state._model_fingerprint != dst.state._model_fingerprint):
    raise ValueError('state copy requires identical compiled models')
  src_ids, ns = _selected_ids(src, src_env_ids)
  dst_ids, nd = _selected_ids(dst, dst_env_ids)
  if ns != nd:
    raise ValueError('source and destination must select the same world count')
  # Resolve the actual indexed device; 'mps' metadata and tensor 'mps:0'
  # denote the same runtime, but torch.device equality distinguishes them.
  device = dst.state._qpos.device
  if src.state._qpos.device != device:
    raise ValueError('state copy requires the same device')
  fields = []
  temporary_bytes = 0
  for bit, (name, attr, _) in _STATE_FIELDS.items():
    if not spec & bit:
      continue
    sv, dv = _backing(src, attr), _backing(dst, attr)
    ss, ds = _shape(sm, attr, src.batch_size), _shape(dm, attr, dst.batch_size)
    width = int(np.prod(ss[1:], dtype=np.int64))
    if ss[1:] != ds[1:]:
      raise ValueError(f'{name} source and destination layouts differ')
    if width == 0:
      for value, shape in ((sv, ss), (dv, ds)):
        if value is not None and (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape):
          raise ValueError(f'{name} has incompatible empty state backing')
      continue
    if not isinstance(sv, torch.Tensor) or not isinstance(dv, torch.Tensor):
      raise ValueError(f'{name} has no owned state backing')
    dtype = torch.int32 if attr == '_eq_active' else torch.float32
    if (tuple(sv.shape) != ss or tuple(dv.shape) != ds or sv.dtype != dtype or dv.dtype != dtype or
        sv.device != device or dv.device != device or
        not sv.is_contiguous() or not dv.is_contiguous()):
      raise ValueError(f'{name} has incompatible state backing')
    if torch.is_inference(dv):
      raise ValueError(f'{name} has immutable inference destination backing')
    fields.append((name, attr, sv, dv))
    temporary_bytes += 2 * nd * width * dv.element_size()
  if not fields:
    return None
  precision = []
  for bit, attr, position in ((StateSpec.QPOS, '_qpos', True),
                              (StateSpec.QVEL, '_qvel', False)):
    if not spec & bit or not _shape(sm, attr, 1)[1]:
      continue
    sp, dp = src.state._precision, dst.state._precision
    if (sp.nq, sp.nv) != (dp.nq, dp.nv):
      raise ValueError('retained state layouts differ')
    shadow = dp.qpos_shadow if position else dp.qvel_shadow
    if torch.is_inference(dp.words) or torch.is_inference(shadow):
      raise ValueError('retained state has immutable inference destination backing')
    precision.append((sp, dp, attr, position))
    width = 2 * (dp.nq if position else dp.nv)
    # Private reconciliation, saved words and shadows, staged selected words.
    temporary_bytes += 4 * (src.batch_size * width +
                            3 * dst.batch_size * width + nd * width)
  # Charge explicit boundary staging conservatively before allocating it.
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  from mujoco_metal.native_api import _owned_native_backing_bytes
  limit = (getattr(dst, 'limits', None) or CapacityLimits()).memory_budget_bytes
  temporary_bytes += 8 * (ns + nd)
  if _owned_native_backing_bytes(dst) + temporary_bytes > limit:
    raise CapacityOverflow('selected state copy exceeds owned memory budget')
  si = None if src_ids is None else torch.as_tensor(src_ids, dtype=torch.long, device=device)
  di = None if dst_ids is None else torch.as_tensor(dst_ids, dtype=torch.long, device=device)
  def rows(value, ids):
    return value.clone() if ids is None else value.index_select(0, ids)
  staged = [(name, dv, rows(sv, si), rows(dv, di)) for name, _, sv, dv in fields]
  word_stages = []
  for sp, dp, attr, position in precision:
    sw, _ = _effective_precision(sp, _backing(src, attr), position)
    dw, span = _effective_precision(dp, _backing(dst, attr), position)
    shadow = dp.qpos_shadow if position else dp.qvel_shadow
    original = dp._qpos if position else dp._qvel
    word_stages.append((dp, attr, position, span, dw, rows(sw, si),
                        dp.words[:, span].clone(), shadow, shadow.clone(), original))
  metadata = _copy_metadata(dst)
  def write(target, value):
    if di is None:
      target.copy_(value)
    else:
      target.index_copy_(0, di, value)
  with torch.no_grad():
    try:
      for _, target, value, _ in staged:
        write(target, value)
      for dp, attr, position, span, words, selected, _, shadow, _, _ in word_stages:
        write(words, selected)
        dp.words[:, span].copy_(words)
        shadow.copy_(_backing(dst, attr))
        if position:
          dp._qpos = _backing(dst, attr)
        else:
          dp._qvel = _backing(dst, attr)
      _invalidate(dst)
    except Exception:
      for _, target, _, saved in staged:
        write(target, saved)
      for dp, _, position, span, _, _, saved, shadow, old_shadow, original in word_stages:
        dp.words[:, span].copy_(saved)
        shadow.copy_(old_shadow)
        if position:
          dp._qpos = original
        else:
          dp._qvel = original
      values, stages, stage_values, record, coherent, generation, caches = metadata
      for name, value in values.items():
        setattr(dst, name, value)
      if stages is not None:
        vars(stages).clear()
        vars(stages).update(stage_values)
      if record is not None:
        for name, value in coherent:
          record.input_versions[name] = stages._version(value)
      dst.state._generation = generation
      for owner, attrs in caches:
        for name, value in attrs.items():
          setattr(owner, name, value)
      raise
  return None
