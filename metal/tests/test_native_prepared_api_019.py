# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public prepared API contracts using CPU producer stand-ins.

These execute the real coordinator, public wrappers, forward-skip dispatcher,
and unconstrained acceleration orchestration. Native producers are deliberately
not replaced in production; opt-in native tests cover their numerical results.
"""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def _simulation():
  torch = pytest.importorskip('torch')
  from mujoco_metal.forward_stages import ForwardStage, ForwardStageCoordinator
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><body><joint name="j" type="slide" damping=".4"/>
      <geom size=".1" mass="2"/></body></worldbody>
    <actuator><motor joint="j" gear="3"/></actuator>
  </mujoco>''')
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
  data = mujoco.MjData(model)
  data.qpos[:] = .2
  data.qvel[:] = .7
  tensor = lambda x: torch.tensor(np.asarray(x).copy()[None], dtype=torch.float32)

  class Simulation:
    forward_skip = MetalSimulation.forward_skip
    inverse_skip = MetalSimulation.inverse_skip
    step1 = MetalSimulation.step1
    step2 = MetalSimulation.step2
    compare_forward_inverse = MetalSimulation.compare_forward_inverse
    _inverse_discrete_with_preserved_actuator_scratch = (
        MetalSimulation._inverse_discrete_with_preserved_actuator_scratch)
    _inverse_discrete_acceleration = MetalSimulation._inverse_discrete_acceleration
    _inverse_discrete_mass_pair_solve = MetalSimulation._inverse_discrete_mass_pair_solve
    _inverse_discrete_mass_pair_product = MetalSimulation._inverse_discrete_mass_pair_product
    validate_forward_stage_record = MetalSimulation.validate_forward_stage_record
    prepare_forward_acceleration = MetalSimulation.prepare_forward_acceleration

    def __init__(self):
      self._mjmodel = self.model = model
      self.batch_size = 1
      self._state = self.state = SimpleNamespace(
          generation=0, _device=torch.device('cpu'), _torch=torch, _nmocap=0,
          _qpos=tensor(data.qpos), _qvel=tensor(data.qvel), _qacc=tensor(data.qacc),
          _warning_number=torch.zeros((1, 7), dtype=torch.int32),
          _warning_lastinfo=torch.zeros((1, 7), dtype=torch.int32),
          _mpos=torch.zeros(1, 0, 3), _mquat=torch.zeros(1, 0, 4),
          _time=torch.zeros(1))
      self._forward_stages = ForwardStageCoordinator(self)
      self.calls = []
      self._rhs = torch.zeros(1, 1)
      # The CPU producer stand-in has no double residual to contribute, but it
      # implements the current ACC ABI explicitly instead of omitting it.
      self._rhs_low = torch.zeros_like(self._rhs)
      self._applied_force = torch.tensor([[.5]])
      self._control = torch.tensor([[.8]])
      self._component_mass_enabled = False
      self._passive = SimpleNamespace(
          _damp=torch.tensor(model.dof_damping.copy(), dtype=torch.float32),
          _dpoly=torch.tensor(model.dof_dampingpoly.copy(), dtype=torch.float32))
      self._body_wrench = torch.empty(1, 0, 6)
      self._contact = self._joint_constraints = self._coupled_constraints = None
      self._sensors = self._sensordata = None
      self._energy = None
      def solve_dense(M, rhs, **_):
        return (torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1),
                torch.zeros(1, dtype=torch.int32))

      def solve_dense_pair(M, rhs_hi, rhs_low, **kwargs):
        solution, status = solve_dense(M, rhs_hi, **kwargs)
        # This CPU stand-in's declared RHS low plane is exactly zero in every
        # fixture; keep the production three-result dense-pair ABI explicit.
        if torch.count_nonzero(rhs_low).item() != 0:
          raise AssertionError("CPU stand-in cannot discard a nonzero RHS low")
        return solution, torch.zeros_like(solution), status

      self._solver = SimpleNamespace(run_device=solve_dense,
                                     run_pair_device=solve_dense_pair)

    def _prepare_sleep_schedule(self, *_):
      return None

    def _prepare_control(self, value):
      self._control.copy_(value)

    def prepare_forward_position(self, qpos=None, *, mocap_pos=None,
                                 mocap_quat=None, skipsensor=False):
      self.calls.append('POS')
      qp = self.state._qpos if qpos is None else qpos
      data.qpos[:] = qp.numpy()[0]
      mujoco.mj_fwdPosition(model, data)
      from tests.pinned_pose_abi import pinned_pose_abi
      poses = pinned_pose_abi(data)
      return self._forward_stages.begin(generation=self.state.generation,
          qpos=qp, mocap_pos=self.state._mpos if mocap_pos is None else mocap_pos,
          mocap_quat=self.state._mquat if mocap_quat is None else mocap_quat,
          position={'poses': poses, 'awake_lists': None, 'skipsensor': skipsensor})

    def prepare_forward_velocity(self, record, qvel=None, *, skipsensor=False):
      self.calls.append('VEL')
      qv = self.state._qvel if qvel is None else qvel
      data.qvel[:] = qv.numpy()[0]
      mujoco.mj_fwdVelocity(model, data)
      mass = np.empty((model.nv, model.nv))
      mujoco.mj_fullM(model, data, mass)
      dynamics = {'poses': record.values[ForwardStage.POS]['poses'],
          'mass_matrix': tensor(mass), **{name: tensor(value) for name, value in (
              ('qfrc_bias', data.qfrc_bias), ('cvel', data.cvel),
              ('cdof', data.cdof), ('cdof_dot', data.cdof_dot),
              ('root_com', data.subtree_com))}}
      dynamics['qfrc_bias_low'] = torch.zeros_like(dynamics['qfrc_bias'])
      value = {'qvel': qv, 'dynamics': dynamics, 'qfrc_bias': dynamics['qfrc_bias'],
               'qfrc_passive': tensor(data.qfrc_passive), 'status': None,
               'skipsensor': skipsensor}
      self._forward_stages.publish(record, ForwardStage.VEL, value)
      return value

    def prepare_forward_actuation(self, record, *, ctrl=None):
      self.calls.append('ACT')
      control = self._control if ctrl is None else ctrl
      data.ctrl[:] = control.numpy()[0]
      mujoco.mj_fwdActuation(model, data)
      value = {'qfrc_actuator': tensor(data.qfrc_actuator)}
      self._forward_stages.publish(record, ForwardStage.ACT, value)
      return value

    def prepare_forward_constraint(self, record, *, skipsensor=False):
      self.calls.append('CONSTRAINT')
      acceleration = record.values[ForwardStage.ACC]
      value = {'qacc': acceleration['qacc_smooth'], 'qfrc_constraint': torch.zeros(1, 1),
               'status': acceleration['status'], 'skipsensor': skipsensor}
      self._forward_stages.publish(record, ForwardStage.CONSTRAINT, value)
      return value

  return Simulation(), data


def test_public_stages_reuse_pinned_prefix_and_applied_override():
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import (mj_fwdPosition, mj_fwdVelocity,
      mj_fwdActuation, mj_fwdAcceleration, mj_fwdConstraint)
  sim, data = _simulation()
  qpos, qvel = np.array([[.25]], np.float32), np.array([[-.6]], np.float32)
  poses = mj_fwdPosition(sim, qpos)
  assert sim.calls == ['POS']
  dynamics = mj_fwdVelocity(sim, qpos, qvel, poses=poses)
  assert sim.calls == ['POS', 'VEL']
  assert mj_fwdVelocity(sim, qpos, qvel, poses, dynamics) is dynamics
  assert sim.calls == ['POS', 'VEL']
  force = mj_fwdActuation(sim, qpos, qvel, poses, np.array([[.9]], np.float32))
  assert force.item() == pytest.approx(2.7)
  acc, status = mj_fwdAcceleration(sim, qpos, qvel, poses, dynamics,
                                    np.array([[1.1]], np.float32))
  expected = np.linalg.solve(np.array([[2.]]),
      np.array([1.1]) + data.qfrc_actuator + data.qfrc_passive - data.qfrc_bias)
  np.testing.assert_allclose(acc.numpy()[0], expected, atol=2e-7)
  assert not status.any()
  constrained, _, borrowed = mj_fwdConstraint(sim, qpos, qvel, poses, dynamics)
  assert constrained is acc and borrowed is dynamics
  assert sim.calls == ['POS', 'VEL', 'ACT', 'CONSTRAINT']
  torch.testing.assert_close(sim._applied_force, torch.tensor([[.5]]))
  torch.testing.assert_close(sim.state._qpos, torch.tensor([[.2]]))
  torch.testing.assert_close(sim._control, torch.tensor([[.8]]))


def test_foreign_or_changed_inputs_fail_before_any_stage_writes():
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import (mj_fwdPosition, mj_fwdVelocity,
      mj_fwdActuation, mj_fwdAcceleration, mj_fwdConstraint)
  sim, _ = _simulation()
  poses = mj_fwdPosition(sim)
  dynamics = mj_fwdVelocity(sim, poses=poses)
  before = list(sim.calls)
  with pytest.raises(ValueError, match='position-stage record'):
    mj_fwdVelocity(sim, poses=dict(poses))
  with pytest.raises(ValueError, match='velocity-stage record'):
    mj_fwdAcceleration(sim, dynamics=dict(dynamics))
  with pytest.raises(ValueError, match='captured stage input'):
    mj_fwdActuation(sim, qpos=np.array([[.4]], np.float32))
  with pytest.raises(ValueError, match='captured stage input tensor'):
    mj_fwdActuation(sim, qpos=sim.state._qpos.clone())
  with pytest.raises(ValueError, match='needs ACC'):
    mj_fwdConstraint(sim)
  assert sim.calls == before
  with pytest.raises(ValueError, match='finite'):
    mj_fwdAcceleration(sim, qfrc_applied=torch.tensor([[float('nan')]]))
  assert sim.calls == before


@pytest.mark.parametrize('skip', [mujoco.mjtStage.mjSTAGE_POS,
                                  mujoco.mjtStage.mjSTAGE_VEL,
                                  mujoco.mjtStage.mjSTAGE_ACC])
def test_public_forward_skip_consumes_exact_record_and_host_inputs(skip):
  from mujoco_metal.native_api import mj_forward, mj_forwardSkip
  sim, _ = _simulation()
  first = mj_forward(sim)
  sim.calls.clear()
  result = mj_forwardSkip(sim, skip, True, record=first['record'],
      qpos=np.array([[.2]], np.float32), qvel=np.array([[.7]], np.float32))
  assert result['record'] is first['record']
  assert sim.calls == (['VEL', 'ACT', 'CONSTRAINT'] if int(skip) == 1
                       else ['ACT', 'CONSTRAINT'])
  assert result['constraint']['skipsensor'] is True


@pytest.mark.parametrize('skip', [True, 1.2, '1', -1, 4])
def test_public_skip_rejects_invalid_stage_before_production(skip):
  from mujoco_metal.native_api import mj_forwardSkip
  sim, _ = _simulation()
  with pytest.raises((ValueError, TypeError), match='skipstage'):
    mj_forwardSkip(sim, skip)
  assert sim.calls == []


def test_public_stage_requires_current_prefix_and_generation():
  from mujoco_metal.native_api import mj_fwdPosition, mj_fwdVelocity
  sim, _ = _simulation()
  with pytest.raises(ValueError, match='no forward-stage record'):
    mj_fwdVelocity(sim)
  record = mj_fwdPosition(sim, return_record=True)
  sim.state.generation += 1
  with pytest.raises(ValueError, match='stale'):
    mj_fwdVelocity(sim, record=record)
  assert sim.calls == ['POS']


@pytest.mark.parametrize('fail', [False, True])
def test_query_restores_record_owners_values_and_mutation_tokens(fail):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import mj_forward, _inverse_query_workspaces
  from mujoco_metal.forward_stages import ForwardStage
  sim, _ = _simulation()
  result = mj_forward(sim)
  record = result['record']
  original_qpos = record.qpos.clone()
  original_qvel = record.values[ForwardStage.VEL]['qvel'].clone()
  sim._forward_position_epoch = 12
  sim._last_actuation_kin = {'force': torch.tensor([[.7]])}
  owner = sim._last_actuation_kin['force']
  def query():
    with _inverse_query_workspaces(sim):
      record.qpos.add_(9)
      record.values[ForwardStage.VEL]['qvel'].add_(8)
      sim._forward_stages.invalidate()
      sim._forward_position_epoch = 99
      sim._last_actuation_kin['force'] = torch.tensor([[99.]])
      if fail:
        raise RuntimeError('query failed')
  if fail:
    with pytest.raises(RuntimeError, match='query failed'):
      query()
  else:
    query()
  assert sim._forward_position_epoch == 12
  assert sim._last_actuation_kin['force'] is owner
  torch.testing.assert_close(owner, torch.tensor([[.7]]))
  assert sim._forward_stages._record is record
  torch.testing.assert_close(record.qpos, original_qpos)
  torch.testing.assert_close(record.values[ForwardStage.VEL]['qvel'], original_qvel)
  sim.validate_forward_stage_record(record, ForwardStage.CONSTRAINT)


@pytest.mark.parametrize('root', ['_effective_implicit',
                                 '_velocity_derivative_values', '_flex_implicit',
                                 '_sparse_flex_implicit'])
@pytest.mark.parametrize('fail', [False, True])
def test_query_restores_sparse_implicit_storage_owners_after_success_or_failure(root, fail):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  sim, _ = _simulation()
  values = torch.tensor([[.3, -.7]])
  workspace = {'values': values}
  program = SimpleNamespace(_workspace=workspace, values=values)
  setattr(sim, root, program)

  def query():
    with _inverse_query_workspaces(sim):
      values.add_(9)
      program.values = torch.full_like(values, 27)
      workspace['values'] = program.values
      workspace['temporary'] = torch.ones_like(values)
      if fail:
        raise RuntimeError('derivative query failed')

  if fail:
    with pytest.raises(RuntimeError, match='derivative query failed'):
      query()
  else:
    query()
  assert program.values is values
  assert program._workspace is workspace
  assert workspace == {'values': values}
  torch.testing.assert_close(values, torch.tensor([[.3, -.7]]), rtol=0, atol=0)


def test_query_does_not_rehabilitate_an_already_mutated_record():
  from mujoco_metal.native_api import mj_forward, _inverse_query_workspaces
  sim, _ = _simulation()
  record = mj_forward(sim)['record']
  record.qpos.add_(1)
  with _inverse_query_workspaces(sim):
    pass
  with pytest.raises(ValueError, match='mutated'):
    sim.validate_forward_stage_record(record, 'POS')


@pytest.mark.parametrize('skip', [mujoco.mjtStage.mjSTAGE_NONE,
                                  mujoco.mjtStage.mjSTAGE_POS,
                                  mujoco.mjtStage.mjSTAGE_VEL])
def test_public_inverse_skip_stages_inputs_and_restores_query_record(skip):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import mj_forward, mj_inverseSkip
  sim, _ = _simulation()
  record = mj_forward(sim)['record']
  pointer = sim._rhs.data_ptr()
  before = sim._rhs.clone()
  calls = []
  def query(stage, skipsensor, **kwargs):
    calls.append((stage, skipsensor, kwargs))
    sim._rhs.fill_(42)
    sim._forward_stages.invalidate()
    return {'qfrc_inverse': sim._rhs, 'status': torch.zeros(1, dtype=torch.int32),
            'record': object()}
  sim.inverse_skip = query
  output = mj_inverseSkip(sim, skip, True, record=record,
      qpos=np.array([[.2]], np.float32), qvel=np.array([[.7]], np.float32),
      qacc=np.array([[1.3]], np.float32), return_details=True)
  assert calls[0][0] == int(skip) and calls[0][1] is True
  torch.testing.assert_close(calls[0][2]['qacc'], torch.tensor([[1.3]]))
  assert 'record' not in output
  torch.testing.assert_close(output['qfrc_inverse'], torch.tensor([[42.]]))
  assert output['qfrc_inverse'].data_ptr() != pointer
  assert sim._rhs.data_ptr() == pointer
  torch.testing.assert_close(sim._rhs, before)
  sim.validate_forward_stage_record(record, 'CONSTRAINT')


@pytest.mark.parametrize('discrete', [False, True])
@pytest.mark.parametrize('disable_damping', [False, True])
def test_public_inverse_uses_actual_prepared_inverse_and_pinned_euler_flag(
    discrete, disable_damping):
  torch = pytest.importorskip('torch')
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.native_api import mj_forward, mj_inverse
  sim, data = _simulation()
  model = sim._mjmodel
  if discrete:
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  if disable_damping:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
  old = mj_forward(sim)['record']
  old_qp, old_qv = old.qpos.clone(), old.values[ForwardStage.VEL]['qvel'].clone()
  requested = np.array([[1.3]], np.float32)
  qp, qv = np.array([[-.1]], np.float32), np.array([[-.6]], np.float32)
  pointer = sim.state._qacc.data_ptr()
  saved = sim.state._qacc.clone()
  result = mj_inverse(sim, qp, qv, requested)
  # Producers are pinned CPU stand-ins; the API, stage dispatcher, inverse
  # force assembly and discrete conversion above are real production methods.
  data.qpos[:] = qp[0]
  data.qvel[:] = qv[0]
  data.qacc[:] = requested[0]
  mujoco.mj_inverse(model, data)
  np.testing.assert_allclose(result.numpy()[0], data.qfrc_inverse, atol=3e-7)
  assert result.data_ptr() != pointer
  assert sim.state._qacc.data_ptr() == pointer
  torch.testing.assert_close(sim.state._qacc, saved)
  torch.testing.assert_close(old.qpos, old_qp)
  torch.testing.assert_close(old.values[ForwardStage.VEL]['qvel'], old_qv)
  sim.validate_forward_stage_record(old, 'CONSTRAINT')


def test_public_inverse_bad_acceleration_rejects_before_producer_dispatch():
  from mujoco_metal.native_api import mj_inverse
  sim, _ = _simulation()
  with pytest.raises(ValueError, match='qacc must be finite'):
    mj_inverse(sim, qacc=np.array([[np.nan]], np.float32))
  assert sim.calls == []


def test_public_step1_callback_order_and_whole_step2_input_preflight():
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import mj_step1, mj_step2
  from mujoco_metal.forward_stages import ForwardStage
  sim, _ = _simulation()
  with pytest.raises(TypeError, match='control_callback'):
    mj_step1(sim, control_callback='invalid')
  assert sim.calls == []
  with pytest.raises(ValueError, match='requires a successful step1'):
    mj_step2(sim)
  def callback(stage, record):
    assert sim.calls == ['POS', 'VEL'] and record.stage == ForwardStage.VEL
    torch.testing.assert_close(stage._qpos, record.qpos)
    sim.calls.append('CONTROL')
    return torch.tensor([[.9]])
  record = mj_step1(sim, control_callback=callback)
  assert record is sim._step1_record
  torch.testing.assert_close(sim._control, torch.tensor([[.9]]))
  before = list(sim.calls)
  with pytest.raises(ValueError, match='xfrc_applied must be finite'):
    mj_step2(sim, ctrl=np.array([[.1]], np.float32),
             xfrc_applied=np.full((1, sim._mjmodel.nbody, 6), np.nan, np.float32))
  assert sim.calls == before and sim._step1_record is record
  torch.testing.assert_close(sim._control, torch.tensor([[.9]]))


def test_public_comparison_reads_existing_record_and_rejects_foreign_bundle():
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import mj_forward, mj_compareFwdInv
  sim, _ = _simulation()
  forward = mj_forward(sim)
  before = list(sim.calls)
  sim.forward_skip = lambda *a, **k: (_ for _ in ()).throw(
      AssertionError('comparison must not run another forward optimizer'))
  result = mj_compareFwdInv(sim)
  torch.testing.assert_close(result, torch.zeros(1, 2))
  assert sim.calls == before
  foreign = dict(forward, constraint=dict(forward['constraint']))
  with pytest.raises(ValueError, match='forward_result must reference'):
    mj_compareFwdInv(sim, forward_result=foreign)
  assert sim.calls == before
