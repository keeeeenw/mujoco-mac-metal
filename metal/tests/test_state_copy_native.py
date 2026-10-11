"""Native public state-copy physics and replay qualification."""
from dataclasses import fields, is_dataclass
import os
import mujoco
import numpy as np
import pytest

import mujoco_metal.state_copy as implementation
from mujoco_metal.native_api import mj_copyState
from mujoco_metal.native_api import StateSpec, _STATE_FIELDS, mj_getState
from mujoco_metal.forward_stages import ForwardStage


def _bytes_equal(a, b, replay=False):
  if isinstance(a, np.ndarray):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
  elif isinstance(a, dict):
    assert a.keys() == b.keys()
    for name in a:
      _bytes_equal(a[name], b[name], replay)
  elif is_dataclass(a):
    for field in fields(a):
      if replay and field.name == 'row_reset_epoch':
        continue  # Explicit restore advances borrowed row epochs.
      _bytes_equal(getattr(a, field.name), getattr(b, field.name), replay)
  else:
    assert a == b


def _model(sparse):
  spec = mujoco.MjSpec.from_string('''<mujoco><size nuserdata="3"/>
    <option timestep=".0001" solver="PGS" iterations="80"><flag energy="enable"/></option>
    <worldbody><geom type="plane" size="2 2 .1" condim="1"/>
      <body name="load" pos="0 0 .099"><joint name="j" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1" condim="1"/></body>
      <body name="target" mocap="true" pos="0 0 .2"><geom type="sphere" size=".01"/></body>
    </worldbody>
    <equality><connect body1="load" body2="target" anchor="0 0 .1"/></equality>
    <actuator><general name="drive" joint="j" dyntype="filterexact" dynprm=".01"/></actuator>
    <sensor><jointpos name="position" joint="j"/><jointvel joint="j"/>
      <actuatorfrc actuator="drive"/></sensor>
    </mujoco>''')
  for entry in (spec.actuator('drive'), spec.sensor('position')):
    entry.nsample, entry.interp, entry.delay = 4, 1, .0002
  model = spec.compile()
  if sparse:
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  else:
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
  return model


def _simulation_pair(sparse):
  import torch
  from mujoco_metal import MetalSimulation
  model = _model(sparse)
  profile = 'integrated_implicitfast_v1' if sparse else 'integrated_euler_v1'
  src, dst = (MetalSimulation(model, 3, profile=profile) for _ in range(2))
  for index, sim in enumerate((src, dst)):
    qpos = np.array([[-.00234567890123456], [-.00123456789012345],
                     [-.00298765432109876]]) - index * .0002
    qvel = np.array([[.0212345678901234], [-.0152345678901234], [.0312345678901234]])
    mpos = np.tile(np.array([[[0., 0., .2]]]), (3, 1, 1))
    mpos[:, 0, 2] += (index + 1) * .0001
    sim.reset(qpos=qpos, qvel=qvel, act=np.array([[.2], [.3], [.1]]) * (index + 1),
              eq_active=np.array([[1], [0], [1]]), mocap_pos=mpos,
              mocap_quat=np.tile(np.array([[[1., 0., 0., 0.]]]), (3, 1, 1)))
    ctrl = np.array([[.4], [.6], [.5]], dtype=np.float32) * (index + 1)
    applied = np.array([[.025], [-.035], [.045]], dtype=np.float32) * (index + 1)
    wrench = np.zeros((3, model.nbody, 6), dtype=np.float32)
    wrench[:, 1, 2] = np.array([.011, -.012, .013]) * (index + 1)
    assert sim.step(3 + index, ctrl=ctrl, qfrc_applied=applied,
                    xfrc_applied=wrench).cpu().tolist() == [0, 0, 0]
    sim.state._userdata.copy_(torch.arange(9, device=sim.state._device).reshape(3, 3) + index)
  return model, src, dst


def _cpu_data(model, sim):
  values = mj_getState(sim)
  snapshot = sim.snapshot()['device']
  words = snapshot.precision_words.astype(np.float64)
  nq, nv = model.nq, model.nv
  represented_q = snapshot.qpos.astype(float) + words[:, :nq] + words[:, nq:2*nq]
  represented_v = snapshot.qvel.astype(float) + words[:, 2*nq:2*nq+nv] + words[:, 2*nq+nv:]
  mapping = dict(time='time', qpos='qpos', qvel='qvel', act='act', history='history',
                 warmstart='qacc_warmstart', ctrl='ctrl', qfrc_applied='qfrc_applied',
                 xfrc_applied='xfrc_applied', eq_active='eq_active', mocap_pos='mocap_pos',
                 mocap_quat='mocap_quat', userdata='userdata', plugin_state='plugin_state')
  data = []
  for row in range(sim.batch_size):
    d = mujoco.MjData(model)
    for name, field in mapping.items():
      raw = values[name][row].cpu().numpy()
      if name == 'time':
        d.time = float(raw)
      else:
        getattr(d, field)[:] = raw
    d.qpos[:], d.qvel[:] = represented_q[row], represented_v[row]
    data.append(d)
  return data


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1',
                                               reason='native state-copy opt-in')]


@pytest.mark.parametrize('sparse', (False, True))
@pytest.mark.parametrize('mask', [0, *list(_STATE_FIELDS), StateSpec.PHYSICS,
                                StateSpec.USER, StateSpec.FULLPHYSICS, StateSpec.ALL])
def test_all_masks_copy_with_contact_history_equality_and_replay(sparse, mask, monkeypatch):
  import torch
  model, src, dst = _simulation_pair(sparse)
  cpu_src, cpu_dst = _cpu_data(model, src), _cpu_data(model, dst)
  before_src, before_dst = mj_getState(src), mj_getState(dst)
  before_words = dst.state._precision.words.clone()
  source_words = src.state._precision.words.clone()
  source_checkpoint = src.snapshot()
  # Real producer views precede a pending split-step record.
  borrowed = dst.assembled_system(recompute=True)
  dst.step1()
  pending = dst._step1_record
  buffers = [(value, value.clone()) for value in borrowed.values() if isinstance(value, torch.Tensor)]
  unrelated = dst.snapshot()
  for source, destination in ((2, 0), (0, 2)):
    mujoco.mj_copyState(model, cpu_src[source], cpu_dst[destination], mask)
  with monkeypatch.context() as guard:
    def forbid(*args, **kwargs):
      raise AssertionError('copy may not evaluate physics or delegate to CPU')
    for name in ('mj_step', 'mj_forward', 'mj_collision', 'mj_copyState', 'mj_fwdPosition'):
      guard.setattr(mujoco, name, forbid)
    for program in (dst._smooth, dst._passive, dst._coupled_constraints):
      for name in ('run_device', 'run_kinematics', 'generate_candidates'):
        if program is not None and hasattr(program, name):
          guard.setattr(program, name, forbid)
    mj_copyState(src, dst, mask, src_env_ids=[2, 0], dst_env_ids=[0, 2])
  after = mj_getState(dst)
  for bit, (name, _, _) in _STATE_FIELDS.items():
    expected = before_dst[name].clone()
    if mask & bit:
      expected[[0, 2]] = before_src[name][[2, 0]]
    assert expected.cpu().numpy().tobytes() == after[name].cpu().numpy().tobytes()
  expected_words = before_words.clone()
  for bit, span in ((StateSpec.QPOS, slice(0, 2 * model.nq)),
                    (StateSpec.QVEL, slice(2 * model.nq, None))):
    if mask & bit:
      expected_words[[0, 2], span] = source_words[[2, 0], span]
  assert expected_words.cpu().numpy().tobytes() == dst.state._precision.words.cpu().numpy().tobytes()
  _bytes_equal(source_checkpoint, src.snapshot())
  copied = dst.snapshot()
  for name in ('energy', 'sensordata', 'raw_sensordata', 'plugins', 'sleep'):
    if name in unrelated:
      _bytes_equal(unrelated[name], copied[name])
  assert copied['device'].qacc.tobytes() == unrelated['device'].qacc.tobytes()
  assert all(value.cpu().numpy().tobytes() == saved.cpu().numpy().tobytes()
             for value, saved in buffers)
  changed = mask not in (0, StateSpec.PLUGIN)
  if changed:
    with pytest.raises(ValueError, match='stale'):
      dst.validate_forward_stage_record(pending, 'VEL')
    old_context = pending.values[ForwardStage.POS]['context']
    with pytest.raises(ValueError, match='replaced workspace'):
      dst._smooth.run_velocity_device(old_context['smooth'], dst.state._qvel)
    with pytest.raises(ValueError, match='stale'):
      dst._coupled_constraints.run_velocity_device(
          old_context['coupled'], None, None, dst.state._qvel,
          dst.state._qpos, dst.state._qvel)
  else:
    assert dst._step1_record is pending
  for d in cpu_dst:
    mujoco.mj_forward(model, d)
  assert sum(d.ncon for d in cpu_dst) >= 2
  assert any(np.linalg.norm(d.qfrc_constraint) > .01 for d in cpu_dst)
  native = dst.query_assembled_system()
  for name in ('qacc', 'qfrc_constraint'):
    np.testing.assert_allclose(native[name].cpu().numpy(),
                               np.asarray([getattr(d, name) for d in cpu_dst]),
                               rtol=4e-5, atol=6e-6)
  # Source oracle steps run before fallback guards; native never invokes CPU.
  for d in cpu_dst:
    for _ in range(3):
      mujoco.mj_step(model, d)
  with monkeypatch.context() as guard:
    for name in ('mj_step', 'mj_forward', 'mj_collision', 'mj_fwdPosition'):
      guard.setattr(mujoco, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('CPU fallback')))
    if not changed:
      assert dst.step2().cpu().tolist() == [0, 0, 0]
      assert dst.step(2).cpu().tolist() == [0, 0, 0]
    else:
      assert dst.step(3).cpu().tolist() == [0, 0, 0]
  for name in ('qpos', 'qvel', 'act'):
    np.testing.assert_allclose(getattr(dst.state, '_' + name).cpu().numpy(),
                               np.asarray([getattr(d, name) for d in cpu_dst]),
                               rtol=4e-5, atol=6e-6)
  end = dst.snapshot()
  dst.restore(copied)
  mj_copyState(src, dst, mask, src_env_ids=[2, 0], dst_env_ids=[0, 2])
  if not changed:
    # Replay the same split-step route; a checkpoint does not retain a
    # borrowed pending forward record, and full vs split stepping may use
    # different arithmetic despite equivalent high projections.
    dst.step1()
    assert dst.step2().cpu().tolist() == [0, 0, 0]
    assert dst.step(2).cpu().tolist() == [0, 0, 0]
  else:
    assert dst.step(3).cpu().tolist() == [0, 0, 0]
  _bytes_equal(end, dst.snapshot(), replay=True)


@pytest.mark.parametrize('sparse', (False, True))
@pytest.mark.parametrize('position', (False, True))
@pytest.mark.parametrize('replacement', (False, True))
def test_real_source_writes_and_unavailable_counters_keep_owned_precision(sparse, position, replacement):
  import torch
  model, src, dst = _simulation_pair(sparse)
  attr = '_qpos' if position else '_qvel'
  original_words = src.state._precision.words.clone()
  shadow = (src.state._precision.qpos_shadow if position else src.state._precision.qvel_shadow).clone()
  if replacement:
    with torch.inference_mode():
      high = getattr(src.state, attr).clone()
    setattr(src.state, attr, high)
    with pytest.raises(RuntimeError):
      _ = high._version
  else:
    getattr(src.state, attr)[1, 0] += .0001
  high = getattr(src.state, attr).clone()
  mask = StateSpec.QPOS if position else StateSpec.QVEL
  mj_copyState(src, dst, mask)
  span = slice(0, 2 * model.nq) if position else slice(2 * model.nq, None)
  expected = original_words[:, span].clone()
  if replacement:
    expected.zero_()
  else:
    expected[1].zero_()
  assert expected.cpu().numpy().tobytes() == dst.state._precision.words[:, span].cpu().numpy().tobytes()
  assert original_words.cpu().numpy().tobytes() == src.state._precision.words.cpu().numpy().tobytes()
  actual_shadow = src.state._precision.qpos_shadow if position else src.state._precision.qvel_shadow
  assert shadow.cpu().numpy().tobytes() == actual_shadow.cpu().numpy().tobytes()
  assert high.cpu().numpy().tobytes() == getattr(dst.state, attr).cpu().numpy().tobytes()
  oracle = _cpu_data(model, dst)
  for d in oracle:
    for _ in range(3):
      mujoco.mj_step(model, d)
  assert dst.step(3).cpu().tolist() == [0, 0, 0]
  for name in ('qpos', 'qvel', 'act'):
    np.testing.assert_allclose(getattr(dst.state, '_' + name).cpu().numpy(),
                               np.asarray([getattr(d, name) for d in oracle]),
                               rtol=4e-5, atol=6e-6)


@pytest.mark.parametrize('sparse', (False, True))
def test_native_overlap_reads_all_original_rows_before_writing(sparse):
  model, sim, _ = _simulation_pair(sparse)
  before = mj_getState(sim)
  words = sim.state._precision.words.clone()
  oracle = _cpu_data(model, sim)
  original = [mujoco.MjData(model) for _ in oracle]
  for row, source in enumerate(oracle):
    mujoco.mj_copyState(model, source, original[row], StateSpec.ALL)
  for row, source in enumerate((1, 2, 0)):
    mujoco.mj_copyState(model, original[source], oracle[row], StateSpec.ALL)
  mj_copyState(sim, sim, src_env_ids=[0, 1, 2], dst_env_ids=[2, 0, 1])
  for name, value in mj_getState(sim).items():
    assert value.cpu().numpy().tobytes() == before[name][[1, 2, 0]].cpu().numpy().tobytes()
  assert sim.state._precision.words.cpu().numpy().tobytes() == words[[1, 2, 0]].cpu().numpy().tobytes()
  for d in oracle:
    for _ in range(3):
      mujoco.mj_step(model, d)
  assert sim.step(3).cpu().tolist() == [0, 0, 0]
  for name in ('qpos', 'qvel', 'act'):
    np.testing.assert_allclose(getattr(sim.state, '_' + name).cpu().numpy(),
                               np.asarray([getattr(d, name) for d in oracle]),
                               rtol=4e-5, atol=6e-6)


@pytest.mark.parametrize('sparse', (False, True))
def test_actual_post_write_failure_preserves_pending_stage_and_trajectory(sparse, monkeypatch):
  model, src, dst = _simulation_pair(sparse)
  dst.step1()
  record = dst._step1_record
  before = dst.snapshot()
  calls = []
  original = implementation._invalidate
  def fail(sim):
    calls.append(True)
    original(sim)
    raise RuntimeError('native copy failure after writes')
  with monkeypatch.context() as guard:
    guard.setattr(implementation, '_invalidate', fail)
    with pytest.raises(RuntimeError, match='after writes'):
      mj_copyState(src, dst)
  assert calls == [True]
  _bytes_equal(before, dst.snapshot())
  assert dst.validate_forward_stage_record(record, 'VEL') is record
  assert dst.step2().cpu().tolist() == [0, 0, 0]
  assert dst.step(2).cpu().tolist() == [0, 0, 0]
  end = dst.snapshot()
  dst.restore(before)
  dst.step1()
  assert dst.step2().cpu().tolist() == [0, 0, 0]
  assert dst.step(2).cpu().tolist() == [0, 0, 0]
  _bytes_equal(end, dst.snapshot(), replay=True)


@pytest.mark.parametrize('sparse', (False, True))
def test_different_batch_sizes_copy_then_native_trajectory(sparse, monkeypatch):
  from mujoco_metal import MetalSimulation
  model, src, _ = _simulation_pair(sparse)
  dst = MetalSimulation(model, 4, profile=src.profile.name)
  src_cpu, dst_cpu = _cpu_data(model, src), _cpu_data(model, dst)
  before = mj_getState(dst)
  for source, destination in ((2, 3), (0, 1)):
    mujoco.mj_copyState(model, src_cpu[source], dst_cpu[destination], StateSpec.ALL)
  mj_copyState(src, dst, src_env_ids=[2, 0], dst_env_ids=[3, 1])
  for name, value in mj_getState(dst).items():
    assert value[[0, 2]].cpu().numpy().tobytes() == before[name][[0, 2]].cpu().numpy().tobytes()
    assert value[[3, 1]].cpu().numpy().tobytes() == mj_getState(src)[name][[2, 0]].cpu().numpy().tobytes()
  copied = dst.snapshot()
  for d in dst_cpu:
    for _ in range(3):
      mujoco.mj_step(model, d)
  with monkeypatch.context() as guard:
    for name in ('mj_step', 'mj_forward', 'mj_collision', 'mj_fwdPosition'):
      guard.setattr(mujoco, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('CPU fallback')))
    assert dst.step(3).cpu().tolist() == [0, 0, 0, 0]
  for name in ('qpos', 'qvel', 'act'):
    np.testing.assert_allclose(getattr(dst.state, '_' + name).cpu().numpy(),
                               np.asarray([getattr(d, name) for d in dst_cpu]),
                               rtol=4e-5, atol=6e-6)
  end = dst.snapshot()
  dst.restore(copied)
  mj_copyState(src, dst, src_env_ids=[2, 0], dst_env_ids=[3, 1])
  assert dst.step(3).cpu().tolist() == [0, 0, 0, 0]
  _bytes_equal(end, dst.snapshot(), replay=True)


@pytest.mark.parametrize('sparse', (False, True))
def test_literal_nonfinite_bits_and_immutable_destination_admission(sparse):
  import torch
  model, src, dst = _simulation_pair(sparse)
  src.state._qpos[:, 0].view(torch.int32).copy_(torch.tensor(
      [0x7fc00001, -2147483648, 0x7f800000], dtype=torch.int32,
      device=src.state._qpos.device))
  src.state._mquat.fill_(7.)
  before_src = mj_getState(src)
  mj_copyState(src, dst, StateSpec.QPOS | StateSpec.MOCAP_QUAT)
  for name in ('qpos', 'mocap_quat'):
    assert mj_getState(dst)[name].cpu().numpy().tobytes() == before_src[name].cpu().numpy().tobytes()
  # This is literal state copy, not admission to stepping invalid source values.
  _, src, dst = _simulation_pair(sparse)
  with torch.inference_mode():
    dst.state._qpos = dst.state._qpos.clone()
  before = mj_getState(dst)
  version, generation = dst.state._time._version, dst.state._generation
  with pytest.raises(ValueError, match='immutable'):
    mj_copyState(src, dst)
  assert dst.state._time._version == version and dst.state._generation == generation
  for name, value in mj_getState(dst).items():
    assert value.cpu().numpy().tobytes() == before[name].cpu().numpy().tobytes()
