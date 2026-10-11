"""CPU storage and pinned public state-copy contracts."""
from types import SimpleNamespace
import numpy as np
import mujoco
import pytest
import torch

import mujoco_metal.state_copy as implementation
from mujoco_metal.native_api import mj_copyState
from mujoco_metal import MetalSimulation
from mujoco_metal.device_state import DeviceState
from mujoco_metal.forward_stages import ForwardStageCoordinator, ForwardStage
from mujoco_metal.native_api import StateSpec, _STATE_FIELDS, mj_getState
from mujoco_metal.stepping import validate_stepping_profile


def _sim(batch=3, mass=1):
  spec = mujoco.MjSpec.from_string(f'''<mujoco>
    <size nuserdata="3"/><option timestep=".001"/>
    <worldbody>
      <body name="load"><joint name="j" type="slide"/>
        <geom type="sphere" size=".1" mass="{mass}"/></body>
      <body name="target" mocap="true"><geom type="sphere" size=".01"/></body>
    </worldbody>
    <equality><connect body1="load" body2="target" anchor="0 0 0"/></equality>
    <actuator><general name="drive" joint="j" dyntype="filterexact" dynprm=".1"/></actuator>
    <sensor><jointpos name="position" joint="j"/></sensor>
    </mujoco>''')
  for entry in (spec.actuator('drive'), spec.sensor('position')):
    entry.nsample, entry.interp, entry.delay = 4, 1, .002
  model = spec.compile()
  profile = validate_stepping_profile(model, profile='integrated_euler_v1')
  sim = MetalSimulation.__new__(MetalSimulation)
  sim._mjmodel = model
  sim._state = DeviceState(model, profile, batch, device='cpu')
  sim.batch_size = batch
  sim.limits = None
  sim._control = torch.zeros((batch, model.nu))
  sim._applied_force = torch.zeros((batch, model.nv))
  sim._body_wrench = torch.zeros((batch, model.nbody, 6))
  sim._assembled_system_valid = True
  sim._accepted_step = {'unchanged': True}
  sim._last_coupled = {'unchanged': True}
  sim._last_coupled_generation = 0
  sim._spatial_cache_key = ('prior',)
  sim._spatial_kin = {'unchanged': True}
  sim._step1_record = None
  sim._forward_stages = ForwardStageCoordinator(sim)
  return sim


def _fill(sim, offset):
  for _, (_, attr, _) in _STATE_FIELDS.items():
    t = implementation._backing(sim, attr)
    if t is not None and t.numel():
      raw = torch.arange(t.numel(), dtype=torch.float32).reshape(t.shape) + offset
      if attr == '_eq_active':
        raw = raw.remainder(2)
      t.copy_(raw)
  sim.state._precision.bind(sim.state._qpos, sim.state._qvel)
  sim.state._precision.words.copy_(torch.arange(sim.state._precision.words.numel())
                                  .reshape(sim.state._precision.words.shape) * 1e-9)


def _bits(value):
  return value.detach().numpy().tobytes()


@pytest.mark.parametrize('spec', [0, *list(_STATE_FIELDS), StateSpec.PHYSICS,
                                StateSpec.USER, StateSpec.FULLPHYSICS, StateSpec.ALL])
def test_each_bit_and_composite_plain_copy_with_paired_permutations(spec):
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  before_src = mj_getState(src)
  before_dst = mj_getState(dst)
  before_words = dst.state._precision.words.clone()
  source_words = src.state._precision.words.clone()
  owners = {attr: implementation._backing(dst, attr) for _, attr, _ in _STATE_FIELDS.values()}
  mj_copyState(src, dst, spec, src_env_ids=[2, 0], dst_env_ids=[0, 2])
  after = mj_getState(dst)
  for bit, (name, attr, _) in _STATE_FIELDS.items():
    expected = before_dst[name].clone()
    if spec & bit:
      expected[[0, 2]] = before_src[name][[2, 0]]
    assert _bits(after[name]) == _bits(expected)
    assert implementation._backing(dst, attr) is owners[attr]
    assert _bits(mj_getState(src)[name]) == _bits(before_src[name])
  expected_words = before_words.clone()
  for bit, span in ((StateSpec.QPOS, slice(0, 2 * src.state._precision.nq)),
                    (StateSpec.QVEL, slice(2 * src.state._precision.nq, None))):
    if spec & bit:
      expected_words[[0, 2], span] = source_words[[2, 0], span]
  assert _bits(dst.state._precision.words) == _bits(expected_words)
  assert _bits(src.state._precision.words) == _bits(source_words)
  assert dst.state.generation == (0 if spec in (0, StateSpec.PLUGIN) else 1)


def test_same_sim_overlapping_full_batch_permutation_reads_original_sources():
  sim = _sim()
  _fill(sim, 7)
  original = mj_getState(sim)
  words = sim.state._precision.words.clone()
  mj_copyState(sim, sim, src_env_ids=[0, 1, 2], dst_env_ids=[2, 0, 1])
  for name, value in mj_getState(sim).items():
    assert _bits(value) == _bits(original[name][[1, 2, 0]])
  assert _bits(sim.state._precision.words) == _bits(words[[1, 2, 0]])


def test_plain_copy_preserves_nonfinite_payload_and_signed_zero():
  src, dst = _sim(), _sim()
  # Pinned copy has no setter's finite/quaternion admission.
  src.state._qpos[:, 0].view(torch.int32).copy_(torch.tensor(
      [0x7fc00001, -2147483648, 0x7f800000], dtype=torch.int32))
  src.state._mquat.fill_(7.)
  mj_copyState(src, dst, StateSpec.QPOS | StateSpec.MOCAP_QUAT)
  assert _bits(dst.state._qpos) == _bits(src.state._qpos)
  assert _bits(dst.state._mquat) == _bits(src.state._mquat)


@pytest.mark.parametrize('position', (False, True))
@pytest.mark.parametrize('replacement', (False, True))
def test_source_real_mutation_and_replacement_reconciled_without_source_writes(position, replacement):
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  attr = '_qpos' if position else '_qvel'
  if replacement:
    setattr(src.state, attr, getattr(src.state, attr).clone())
  else:
    getattr(src.state, attr)[1, 0] += .25
  original_words = src.state._precision.words.clone()
  original_shadow = (src.state._precision.qpos_shadow if position else
                     src.state._precision.qvel_shadow).clone()
  bit = StateSpec.QPOS if position else StateSpec.QVEL
  mj_copyState(src, dst, bit)
  n = src.state._precision.nq
  span = slice(0, 2 * n) if position else slice(2 * n, None)
  expected = original_words[:, span].clone()
  expected[:] = 0 if replacement else expected
  if not replacement:
    expected[1] = 0
  assert _bits(dst.state._precision.words[:, span]) == _bits(expected)
  assert _bits(src.state._precision.words) == _bits(original_words)
  assert _bits(src.state._precision.qpos_shadow if position else
               src.state._precision.qvel_shadow) == _bits(original_shadow)
  dst.state._precision.synchronize(dst.state._qpos, dst.state._qvel)
  assert _bits(dst.state._precision.words[:, span]) == _bits(expected)


@pytest.mark.parametrize('bad', (-1, 16384, 1.5, True, np.bool_(True)))
def test_bad_signatures_are_atomic(bad):
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  before = mj_getState(dst)
  with pytest.raises(ValueError):
    mj_copyState(src, dst, bad)
  assert all(_bits(value) == _bits(mj_getState(dst)[name]) for name, value in before.items())
  assert dst.state.generation == 0


def test_compatibility_and_bad_selection_rejected_before_writes():
  src, dst = _sim(), _sim(mass=2)
  with pytest.raises(ValueError, match='compiled models'):
    mj_copyState(src, dst)
  dst = _sim()
  for src_ids, dst_ids in (([1], [0, 1]), ([0, 0], [1, 2]), ([], []), ([3], [0])):
    with pytest.raises((ValueError, IndexError)):
      mj_copyState(src, dst, src_env_ids=src_ids, dst_env_ids=dst_ids)
  assert dst.state.generation == 0


def test_failure_after_real_writes_restores_storage_and_coherent_forward_record(monkeypatch):
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  stage = dst._forward_stages
  record = stage.begin(generation=dst.state.generation, qpos=dst.state._qpos,
                       position={'poses': torch.tensor([13.])})
  dst._step1_record = record
  dst._smooth = SimpleNamespace(_position_context_epoch=3)
  dst._coupled_constraints = SimpleNamespace(_position_context_epoch=5, _position_context_valid=True)
  before = mj_getState(dst)
  words = dst.state._precision.words.clone()
  shadow = dst.state._precision.qpos_shadow.clone()
  epoch = stage.epoch
  original = implementation._invalidate
  calls = []
  def fail(sim):
    calls.append(True)
    original(sim)
    raise RuntimeError('post-write invalidation failure')
  monkeypatch.setattr(implementation, '_invalidate', fail)
  with pytest.raises(RuntimeError, match='post-write'):
    mj_copyState(src, dst)
  assert calls == [True]
  assert all(_bits(value) == _bits(mj_getState(dst)[name]) for name, value in before.items())
  assert _bits(dst.state._precision.words) == _bits(words)
  assert _bits(dst.state._precision.qpos_shadow) == _bits(shadow)
  assert dst.state.generation == 0 and stage.epoch == epoch
  assert stage.validate(record, generation=0, minimum=ForwardStage.POS) is record
  assert dst._step1_record is record
  assert dst._smooth._position_context_epoch == 3
  assert dst._coupled_constraints._position_context_epoch == 5
  assert dst._coupled_constraints._position_context_valid


def test_no_forward_or_history_calls_and_unrelated_buffers_untouched(monkeypatch):
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  buffer = torch.tensor([12., 13.])
  dst._energy = buffer
  dst._sensordata = buffer
  dst.state._qacc.fill_(17.)
  def forbid(*args, **kwargs):
    raise AssertionError('copy must not run physics or history')
  dst._smooth = SimpleNamespace(run_device=forbid)
  dst._spatial_tendons = SimpleNamespace(run_kinematics=forbid)
  dst._flex = SimpleNamespace(update_kinematics=forbid)
  dst._sync_delay_from_history = forbid
  for name in ('mj_forward', 'mj_step', 'mj_collision', 'mj_copyState'):
    monkeypatch.setattr(mujoco, name, forbid)
  mj_copyState(src, dst)
  assert dst._energy is buffer and dst._sensordata is buffer
  assert _bits(buffer) == _bits(torch.tensor([12., 13.]))
  assert torch.equal(dst.state._qacc, torch.full_like(dst.state._qacc, 17.))


@pytest.mark.parametrize('spec', [0, *list(_STATE_FIELDS), StateSpec.PHYSICS,
                                StateSpec.USER, StateSpec.FULLPHYSICS, StateSpec.ALL])
def test_actual_pinned_copy_state_oracle_for_every_component(spec):
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  before_src, before_dst = mj_getState(src), mj_getState(dst)
  names = {field[0]: field[1].removeprefix('_') for field in _STATE_FIELDS.values()}
  names.update(warmstart='qacc_warmstart', ctrl='ctrl', qfrc_applied='qfrc_applied',
               xfrc_applied='xfrc_applied', mocap_pos='mocap_pos', mocap_quat='mocap_quat')
  expected = []
  for row in range(3):
    source, destination = mujoco.MjData(src._mjmodel), mujoco.MjData(src._mjmodel)
    for name, field in names.items():
      if name == 'time':
        source.time, destination.time = float(before_src[name][row]), float(before_dst[name][row])
      else:
        getattr(source, field)[:] = before_src[name][row].numpy()
        getattr(destination, field)[:] = before_dst[name][row].numpy()
    mujoco.mj_copyState(src._mjmodel, source, destination, spec)
    expected.append(destination)
  mj_copyState(src, dst, spec)
  for name, field in names.items():
    actual = mj_getState(dst)[name]
    values = np.asarray([getattr(data, field) for data in expected])
    np.testing.assert_array_equal(actual.numpy(), values.astype(actual.numpy().dtype))


def test_different_batch_sizes_and_existing_limits():
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  src, dst = _sim(2), _sim(4)
  _fill(src, 7)
  _fill(dst, 19)
  before = mj_getState(dst)
  mj_copyState(src, dst, src_env_ids=[1], dst_env_ids=[3])
  after = mj_getState(dst)
  for name in after:
    assert _bits(after[name][:3]) == _bits(before[name][:3])
    assert _bits(after[name][3]) == _bits(mj_getState(src)[name][1])
  dst.limits = CapacityLimits(memory_budget_bytes=1)
  before = mj_getState(dst)
  with pytest.raises(CapacityOverflow):
    mj_copyState(src, dst, src_env_ids=[0], dst_env_ids=[2])
  assert all(_bits(value) == _bits(mj_getState(dst)[name]) for name, value in before.items())


@pytest.mark.parametrize('source', (False, True))
def test_incompatible_time_dtype_rejected_atomically(source):
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  (src if source else dst).state._time = (src if source else dst).state._time.to(torch.float64)
  before = mj_getState(dst)
  with pytest.raises(ValueError, match='time.*backing'):
    mj_copyState(src, dst)
  assert all(_bits(value) == _bits(mj_getState(dst)[name]) for name, value in before.items())
  assert dst.state.generation == 0


def test_empty_component_layout_rejected_without_cache_mutation():
  src, dst = _sim(), _sim()
  src.state._plugin_state = torch.empty((2, 0))
  before = mj_getState(dst)
  with pytest.raises(ValueError, match='empty state backing'):
    mj_copyState(src, dst, StateSpec.PLUGIN)
  assert all(_bits(value) == _bits(mj_getState(dst)[name]) for name, value in before.items())
  assert dst.state._generation == 0 and dst._assembled_system_valid


def test_immutable_destination_rejected_before_any_writes_or_record_changes():
  src, dst = _sim(), _sim()
  _fill(src, 7)
  _fill(dst, 19)
  stages = dst._forward_stages
  record = stages.begin(generation=dst.state.generation, qpos=dst.state._qpos,
                        position={'poses': torch.tensor([13.])})
  # A replaced inference destination cannot be written by ordinary device ops.
  # TIME precedes QPOS in the mask: late rejection must not advance its version.
  with torch.inference_mode():
    dst.state._qpos = dst.state._qpos.clone()
  before = mj_getState(dst)
  version = dst.state._time._version
  with pytest.raises(ValueError, match='immutable'):
    mj_copyState(src, dst)
  assert all(_bits(value) == _bits(mj_getState(dst)[name]) for name, value in before.items())
  assert dst.state._time._version == version
  assert stages.epoch == record.epoch and dst.state._generation == 0
