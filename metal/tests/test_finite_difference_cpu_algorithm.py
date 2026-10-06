# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Execute derivative columns on CPU fake simulations, without GPU kernels."""

from types import SimpleNamespace

import mujoco
import math
import numpy as np
import pytest
import torch

from mujoco_metal import finite_difference as fd
from mujoco_metal import native_api
from mujoco_metal.device_state import DeviceState
from mujoco_metal.finite_difference import _inverse_force_delta
from mujoco_metal.forward_stages import ForwardStage
from mujoco_metal.stepping import validate_stepping_profile


class _Model:
  nq = nv = 1
  na = nu = nsensordata = 1
  nhistory = 0
  nM = nC = 1
  njnt = 1
  opt = SimpleNamespace(integrator=int(mujoco.mjtIntegrator.mjINT_EULER),
                        noslip_iterations=0, enableflags=0, disableflags=0)
  actuator_ctrllimited = np.asarray([1], np.int32)
  actuator_ctrlrange = np.asarray([[-1.0, 1.0]], np.float64)
  dof_Madr = np.asarray([0], np.int32)
  dof_parentid = np.asarray([-1], np.int32)


class _State:
  __module__ = "mujoco_metal.fake_fd_state"

  def __init__(self, batch, nq=1, nv=1, na=1):
    self._qpos = torch.full((batch, nq), .2, dtype=torch.float32)
    self._qvel = torch.linspace(.1, .2, batch, dtype=torch.float32)[:, None]
    self._qacc = torch.full((batch, nv), .3, dtype=torch.float32)
    self._act = (torch.full((batch, na), .4, dtype=torch.float32)
                 if na else None)
    self._status = torch.zeros((batch,), dtype=torch.int32)
    self._generation = 7
    self.device = torch.device("cpu")


class _Sim:
  __module__ = "mujoco_metal.fake_fd_sim"

  def __init__(self, batch=2, *, limited=True, na=1, nu=1, ns=1):
    self._mjmodel = _Model()
    # Options belong to an individual compiled model in production. Avoid
    # sharing the class-level stub across tests that toggle disable flags.
    self._mjmodel.opt = SimpleNamespace(
        integrator=int(mujoco.mjtIntegrator.mjINT_EULER),
        noslip_iterations=0, enableflags=0, disableflags=0)
    self._mjmodel.na = na
    self._mjmodel.nu = nu
    self._mjmodel.nsensordata = ns
    self._mjmodel.actuator_ctrllimited = np.asarray([int(limited)] * nu, np.int32)
    self._mjmodel.actuator_ctrlrange = np.broadcast_to(
        np.asarray([[-1.0, 1.0]]), (nu, 2)).copy()
    self.batch_size = batch
    self._state = _State(batch, na=na)
    self._control = torch.full((batch, nu), .25, dtype=torch.float32)
    self._sensordata = torch.zeros((batch, ns), dtype=torch.float32) if ns else None
    self._prepared_sensor_flags = []
    self._native_plugins = ()
    self._forward_stages = SimpleNamespace(_record=None)
    self._assembled_system_valid = True
    self._inverse_records = []
    self.limits = SimpleNamespace(memory_budget_bytes=1 << 20)
    self.fail_positive_qpos_world0 = False
    self.qpos_slope = .25
    self.act_slope = .5
    self.ctrl_slope = .75
    self._owner = self

  @property
  def device(self):
    # Production Simulation exposes a device string, while its owned tensors
    # expose torch.device; FD validation must follow the latter.
    return "cpu"

  def _invalidate_forward_stage_record(self):
    self._forward_stages._record = None

  def step(self):
    qpos, qvel = self._state._qpos, self._state._qvel
    act = self._state._act
    ctrl = self._control
    if int(self._mjmodel.nu) and bool(self._mjmodel.actuator_ctrllimited[0]):
      ctrl_used = torch.clamp(ctrl, -1, 1)
    else:
      ctrl_used = ctrl
    if self._mjmodel.na:
      qvel.add_(self.act_slope * act)
      act.mul_(.8)
    if self._mjmodel.nu:
      qvel.add_(self.ctrl_slope * ctrl_used)
    qpos.add_(self.qpos_slope * qvel)
    if self._sensordata is not None:
      self._sensordata.copy_(qpos + 2 * qvel)
    if self.fail_positive_qpos_world0:
      failed = (qpos[:, 0] > .20005) & (torch.arange(self.batch_size) == 0)
      self._state._status.copy_(failed.to(torch.int32))
    self._state._generation += 1
    return self._state._status

  def inverse_skip(self, skipstage=mujoco.mjtStage.mjSTAGE_NONE,
                   skipsensor=False, *, return_components=False,
                   sensor_qacc_low=None, record=None):
    self._inverse_records.append(record)
    if record is not None:
      assert ForwardStage.POS in record.values
      if int(skipstage) == int(mujoco.mjtStage.mjSTAGE_VEL):
        assert ForwardStage.VEL in record.values
    q, v, a = self._state._qpos, self._state._qvel, self._state._qacc
    mass = (2 + q).reshape(self.batch_size, 1, 1)
    mass_qacc = mass[:, 0, 0, None] * a
    bias = 3 * q
    passive = -4 * v
    constraint = torch.zeros_like(mass_qacc)
    force = bias + (mass_qacc - passive - constraint)
    sensor_disabled = bool(
        int(self._mjmodel.opt.disableflags)
        & int(mujoco.mjtDisableBit.mjDSBL_SENSOR))
    if self._sensordata is not None and not skipsensor and not sensor_disabled:
      low = (torch.zeros_like(a) if sensor_qacc_low is None
             else sensor_qacc_low)
      self._sensordata.copy_(q + 2 * v + 3 * (a + low))
    dynamics = {"mass_matrix": mass}
    record = SimpleNamespace(values={ForwardStage.VEL: {"dynamics": dynamics}})
    result = {"qfrc_inverse": force, "status": self._state._status.clone(),
              "record": record}
    if return_components:
      result["qfrc_inverse_components"] = {
          "mass_qacc": mass_qacc,
          "bias": bias,
          "bias_low": torch.zeros_like(bias),
          "passive": passive,
          "constraint": constraint,
      }
    return result

  def prepare_forward_position(self, *, skipsensor=False):
    self._prepared_sensor_flags.append(bool(skipsensor))
    record = SimpleNamespace(values={ForwardStage.POS: {}})
    self._prepared_record = record
    return record

  def prepare_forward_velocity(self, record, qvel=None, *, skipsensor=False):
    del qvel
    self._prepared_sensor_flags.append(bool(skipsensor))
    stage_values = {"dynamics": {}}
    record.values[ForwardStage.VEL] = stage_values
    return stage_values


@pytest.fixture
def cpu_query(monkeypatch):
  position_program_calls = []
  monkeypatch.setattr(native_api, "_position_query_program",
                      lambda sim: position_program_calls.append(sim))

  def integrate(sim, qpos, qvel, dt):
    return {"qpos": qpos + qvel * dt,
            "status": torch.zeros((sim.batch_size,), dtype=torch.int32,
                                   device=qpos.device)}

  def differentiate(sim, first, second, dt):
    return {"qvel": (second - first) / dt,
            "status": torch.zeros((sim.batch_size,), dtype=torch.int32,
                                   device=first.device)}

  monkeypatch.setattr(native_api, "mj_integratePos", integrate)
  monkeypatch.setattr(native_api, "mj_differentiatePos", differentiate)
  monkeypatch.setattr(native_api, "mj_fullM", lambda sim, *, dynamics=None:
                      dynamics["mass_matrix"].clone())
  return position_program_calls


def test_step_fd_forward_and_centered_outputs_and_empty_sensors(cpu_query):
  sim = _Sim()
  out = fd.mjd_stepFD(sim, eps=1e-3, flg_centered=False)
  np.testing.assert_allclose(out["DyDq"][:, 0, :].numpy(), [[1, 0, 0]] * 2, atol=2e-4)
  np.testing.assert_allclose(out["DyDv"][:, 0, :].numpy(), [[.25, 1, 0]] * 2, atol=2e-4)
  np.testing.assert_allclose(out["DyDa"][:, 0, :].numpy(), [[.125, .5, .8]] * 2, atol=2e-4)
  np.testing.assert_allclose(out["DyDu"][:, 0, :].numpy(), [[.1875, .75, 0]] * 2, atol=2e-4)
  np.testing.assert_allclose(out["DsDq"][:, 0, 0].numpy(), [1, 1], atol=2e-4)
  np.testing.assert_allclose(out["DsDv"][:, 0, 0].numpy(), [2.25, 2.25], atol=2e-4)
  np.testing.assert_allclose(out["DsDa"][:, 0, 0].numpy(), [1.125, 1.125], atol=2e-4)
  np.testing.assert_allclose(out["DsDu"][:, 0, 0].numpy(), [1.6875, 1.6875], atol=2e-4)
  centered = fd.mjd_stepFD(sim, eps=1e-3, flg_centered=True)
  torch.testing.assert_close(centered["DyDq"], out["DyDq"], atol=2e-4, rtol=0)
  torch.testing.assert_close(centered["DyDu"], out["DyDu"], atol=2e-4, rtol=0)

  empty = _Sim(batch=1, limited=False, na=0, nu=0, ns=0)
  empty_out = fd.mjd_stepFD(empty, eps=1e-3)
  assert empty_out["DyDa"].shape == (1, 0, 2)
  assert empty_out["DyDu"].shape == (1, 0, 2)
  assert empty_out["DsDq"].shape == (1, 1, 0)
  assert empty_out["DsDu"].shape == (1, 0, 0)


def test_step_fd_world_local_limit_stencil_and_disabled_clamping(cpu_query):
  limited = _Sim(batch=2, limited=True)
  limited._control[:, 0] = torch.tensor([1., .5])
  result = fd.mjd_stepFD(limited, eps=1e-3, flg_centered=True)
  np.testing.assert_allclose(result["DyDu"][:, 0, 1].numpy(), [.75, .75], atol=2e-4)
  np.testing.assert_allclose(result["DsDu"][:, 0, 0].numpy(), [1.6875, 1.6875], atol=2e-4)
  unlimited = _Sim(batch=1, limited=False)
  unlimited._control.fill_(1.)
  result = fd.mjd_stepFD(unlimited, eps=1e-3, flg_centered=True)
  np.testing.assert_allclose(result["DyDu"][:, 0, 1].numpy(), [.75], atol=2e-4)


def test_step_fd_failed_world_is_zero_and_healthy_world_continues(cpu_query):
  sim = _Sim(batch=2)
  sim.fail_positive_qpos_world0 = True
  result = fd.mjd_stepFD(sim, eps=1e-3, flg_centered=False)
  assert result["status"].tolist() == [1, 0]
  assert torch.count_nonzero(result["DyDq"][0]) == 0
  torch.testing.assert_close(result["DyDq"][1, 0], torch.tensor([1., 0., 0.]),
                             atol=2e-4, rtol=0)


def test_inverse_force_delta_preserves_small_constituent_changes():
  base = {
      "mass_qacc": torch.tensor([[0.]], dtype=torch.float32),
      "bias": torch.tensor([[100000000.]], dtype=torch.float32),
      "bias_low": torch.tensor([[0.]], dtype=torch.float32),
      "passive": torch.tensor([[0.]], dtype=torch.float32),
      "constraint": torch.tensor([[0.]], dtype=torch.float32),
  }
  candidate = {key: value.clone() for key, value in base.items()}
  candidate["bias_low"].fill_(1.)
  # Independently rounded force samples cannot resolve +1 at this bias scale,
  # but constituent differencing keeps the requested FD numerator.
  rounded_base = base["bias"] + base["mass_qacc"] + base["bias_low"]
  rounded_candidate = (candidate["bias"] + candidate["mass_qacc"] +
                       candidate["bias_low"])
  torch.testing.assert_close(rounded_candidate - rounded_base,
                             torch.zeros_like(rounded_base))
  delta = _inverse_force_delta(base, candidate)
  torch.testing.assert_close(delta, torch.ones_like(delta), rtol=0, atol=0)

  base_act = torch.tensor([[3.]], dtype=torch.float32)
  candidate_act = torch.tensor([[3.25]], dtype=torch.float32)
  delta = _inverse_force_delta(base, candidate, base_actuation=base_act,
                              candidate_actuation=candidate_act)
  torch.testing.assert_close(delta, torch.full_like(delta, .75),
                             rtol=0, atol=0)


def test_compensated_bias_dot_words_track_binary64_reference():
  # CPU transcription of the Metal two-word product/sum used by smooth_bias.
  # Product residuals are exact for float32 operands and the two_sum sequence
  # must retain the low word even when the legacy high word is unchanged.
  def two_sum(a, b):
    total = np.float32(a + b)
    b_part = np.float32(total - a)
    error = np.float32(
        np.float32(a - np.float32(total - b_part)) + np.float32(b - b_part))
    return total, error

  def add(acc, value):
    hi, lo = acc
    total, err = two_sum(hi, value)
    tail = np.float32(lo + err)
    new_hi = np.float32(total + tail)
    new_lo = np.float32(tail - np.float32(new_hi - total))
    return new_hi, new_lo

  def add_product(acc, a, b):
    product = np.float32(a * b)
    product_error = np.float32(float(a) * float(b) - float(product))
    return add(add(acc, product), product_error)

  left = np.asarray([1.125, -3.75, .1, 4.0, -2.5, .003], dtype=np.float32)
  right = np.asarray([.125, 2.0, -4.0, 1.5, .75, -8.0], dtype=np.float32)
  acc = (np.float32(0), np.float32(0))
  legacy = np.float32(0)
  for a, b in zip(left, right):
    legacy = np.float32(legacy + np.float32(a * b))
    acc = add_product(acc, a, b)
  corrected_low = np.float32(np.float32(acc[0] - legacy) + acc[1])
  reference = math.fsum(float(a) * float(b) for a, b in zip(left, right))
  legacy_high = np.float32(0)
  for a, b in zip(left, right):
    legacy_high = np.float32(legacy_high + np.float32(a * b))
  assert legacy == legacy_high
  assert abs(float(legacy) + float(corrected_low) - reference) <= 1e-7


def test_paired_spatial_bias_accumulators_retain_sub_ulp_stage_deltas():
  """CPU DD transcription checks the upstream spatial product path."""
  def two_sum(left, right):
    total = np.float32(left + right)
    right_part = np.float32(total - left)
    error = np.float32(
        np.float32(left - np.float32(total - right_part)) +
        np.float32(right - right_part))
    return total, error

  def add_words(left, right):
    total, error = two_sum(left[0], right[0])
    tail = np.float32(error + left[1] + right[1])
    high = np.float32(total + tail)
    return high, np.float32(tail - np.float32(high - total))

  def add_product(acc, left, right):
    product = np.float32(left[0] * right[0])
    residual = np.float32(float(left[0]) * float(right[0]) - float(product))
    residual = np.float32(residual + left[0] * right[1] + left[1] * right[0]
                          + left[1] * right[1])
    return add_words(acc, (product, residual))

  def spatial_row(matrix, vector):
    legacy = np.float32(0)
    compensated = (np.float32(0), np.float32(0))
    for coefficient, value in zip(matrix, vector):
      legacy = np.float32(legacy + np.float32(coefficient * value[0]))
      compensated = add_product(
          compensated, (np.float32(coefficient), np.float32(0)), value)
    return legacy, compensated

  # A parent acceleration and three small joint contributions form cacc.
  # The legacy path loses the last contribution; DD preserves it in its low
  # word before the inertia multiply and downstream force projection.
  qvels = np.asarray([.5, -.25, .125], dtype=np.float32)
  cdof_dot = np.asarray([1.0e-7, -2.0e-7, 4.0e-8], dtype=np.float32)
  plus_dot = cdof_dot.copy()
  plus_dot[2] = np.nextafter(plus_dot[2], np.float32(np.inf))

  def accumulate_cacc(dot):
    high = np.float32(2.0e7)
    words = (high, np.float32(0))
    for a, v in zip(dot, qvels):
      high = np.float32(high + np.float32(a * v))
      words = add_product(words, (a, np.float32(0)), (v, np.float32(0)))
    return high, words

  base_high, base_words = accumulate_cacc(cdof_dot)
  plus_high, plus_words = accumulate_cacc(plus_dot)
  assert base_high == plus_high
  stage_delta = math.fsum((float(plus_words[0]), float(plus_words[1]),
                           -float(base_words[0]), -float(base_words[1])))
  source_delta = math.fsum(
      float(a) * float(v) for a, v in zip(plus_dot - cdof_dot, qvels))
  assert abs(stage_delta - source_delta) <= 2e-12

  # Expand the cacc DD word through the inertia row.  The high output still
  # follows the old float32 accumulation while the paired result carries the
  # sub-ULP derivative into the qfrc_bias projection.
  base_force_high, base_force_words = spatial_row(
      np.asarray([1.25], dtype=np.float32), [base_words])
  plus_force_high, plus_force_words = spatial_row(
      np.asarray([1.25], dtype=np.float32), [plus_words])
  assert base_force_high == plus_force_high
  force_delta = math.fsum((float(plus_force_words[0]),
                            float(plus_force_words[1]),
                            -float(base_force_words[0]),
                            -float(base_force_words[1])))
  assert abs(force_delta - 1.25 * source_delta) <= 2e-12


def test_invalid_epsilon_and_overflow_dimensions_fail_before_query_allocation(cpu_query):
  sim = _Sim(batch=2)
  state = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in (
      sim._state._qpos, sim._state._qvel, sim._state._act,
      sim._control, sim._sensordata, sim._state._generation))
  for eps in (0, -1e-3, float("nan"), float("inf"), 1e100, 1e-100):
    with pytest.raises(ValueError, match="eps"):
      fd.mjd_stepFD(sim, eps=eps)
  assert cpu_query == []
  sim._mjmodel.nu = 1
  sim._mjmodel.nv = 50_000
  with pytest.raises(OverflowError, match="int32"):
    fd.mjd_stepFD(sim)
  with pytest.raises(OverflowError, match="int32"):
    fd.mjd_transitionFD(sim)
  assert cpu_query == []
  sim._mjmodel.nv = 1
  sim._mjmodel.nu = (1 << 31)
  with pytest.raises(OverflowError, match="int32"):
    fd.mjd_stepFD(sim)
  assert cpu_query == []
  for actual, expected in zip((sim._state._qpos, sim._state._qvel,
                              sim._state._act, sim._control,
                              sim._sensordata, sim._state._generation), state):
    if isinstance(actual, torch.Tensor):
      torch.testing.assert_close(actual, expected)
    else:
      assert actual == expected


def test_failed_memory_admission_preserves_query_cache_and_simulation_bytes(cpu_query):
  sim = _Sim(batch=2)
  sentinel_cache = SimpleNamespace(token=1, required_bytes=0)
  sim._position_queries = sentinel_cache
  before = [tensor.clone() for tensor in (
      sim._state._qpos, sim._state._qvel, sim._state._act,
      sim._control, sim._sensordata)]
  generation = sim._state._generation
  sim.limits.memory_budget_bytes = 1
  with pytest.raises(ValueError, match="memory budget"):
    fd.mjd_stepFD(sim)
  assert sim._position_queries is sentinel_cache
  assert cpu_query == []
  for actual, expected in zip((sim._state._qpos, sim._state._qvel,
                               sim._state._act, sim._control,
                               sim._sensordata), before):
    torch.testing.assert_close(actual, expected)
  assert sim._state._generation == generation


def test_step_fd_program_and_workspace_budget_exact_edge(cpu_query):
  sim = _Sim(batch=2)
  b, nq, nv, na, nu, ns = 2, 1, 1, 1, 1, 1
  ndx = 2 * nv + na
  extra = (b * (2 * nv + na + nu) * ndx
           + b * (2 * nv + na + nu) * ns + b) * 4
  extra += b * (2 * nq + 8 * nv + 4 * na + nu + 7 * ns) * 4
  required = (fd.estimate_transaction_bytes(sim) + extra
              + fd._position_program_bytes(sim))
  before = sim._state._qpos.clone()
  sim.limits.memory_budget_bytes = required - 1
  with pytest.raises(ValueError, match="memory budget"):
    fd.mjd_stepFD(sim)
  assert cpu_query == []
  torch.testing.assert_close(sim._state._qpos, before)
  sim.limits.memory_budget_bytes = required
  result = fd.mjd_stepFD(sim)
  assert cpu_query == [sim]
  assert result["DyDq"].shape == (b, nv, ndx)


def test_unbounded_plugin_is_rejected_before_query_and_state_mutation(cpu_query):
  sim = _Sim(batch=1)
  sim._native_plugins = (object(),)
  before = sim._state._qpos.clone()
  with pytest.raises(TypeError, match="device_snapshot_bytes"):
    fd.mjd_stepFD(sim)
  assert cpu_query == []
  torch.testing.assert_close(sim._state._qpos, before)


def test_nonfinite_difference_marks_and_zeroes_the_entire_world_row():
  outputs = {
      "DyDq": torch.tensor([[[float("inf"), 2.]], [[3., 4.]]]),
      "DsDq": torch.tensor([[[5.]], [[6.]]]),
      "status": torch.zeros((2,), dtype=torch.int32),
  }
  fd._zero_failed_output_rows(outputs)
  assert outputs["status"].tolist() == [2, 0]
  assert torch.count_nonzero(outputs["DyDq"][0]) == 0
  assert torch.count_nonzero(outputs["DsDq"][0]) == 0
  torch.testing.assert_close(outputs["DyDq"][1], torch.tensor([[3., 4.]]))


def test_sensor_low_word_is_differenced_before_epsilon_division(monkeypatch):
  sim = SimpleNamespace(batch_size=1, _mjmodel=_Model(),
                        device=torch.device("cpu"))
  monkeypatch.setattr(
      fd, "_state_diff",
      lambda _sim, _first, _second, _h, centered=False:
          (torch.zeros((1, 3)), torch.zeros((1,), dtype=torch.int32)))
  status = torch.zeros((1,), dtype=torch.int32)
  state = torch.zeros((1, 1), dtype=torch.float32)
  # All high sensor words are the same rounded large value.  Only the
  # residuals retain the small stencil signal, which must be combined after
  # subtracting each sample rather than adding it back to the high word first.
  baseline = (status, state, state, state, torch.full((1, 1), 259.),
              torch.tensor([[0.]]))
  plus = (status, state, state, state, torch.full((1, 1), 259.),
          torch.tensor([[2e-5]]))
  minus = (status, state, state, state, torch.full((1, 1), 259.),
           torch.tensor([[-2e-5]]))
  _, centered_sensor, _ = fd._difference_candidates(
      sim, baseline, plus, minus, 1e-4, True)
  torch.testing.assert_close(centered_sensor, torch.tensor([[.2]]),
                             atol=1e-6, rtol=0)
  _, forward_sensor, _ = fd._difference_candidates(
      sim, baseline, plus, minus, 1e-4, False)
  torch.testing.assert_close(forward_sensor, torch.tensor([[.2]]),
                             atol=1e-6, rtol=0)


def test_pinned_fd_restrictions_reject_only_source_unsupported_modes(cpu_query):
  sim = _Sim(batch=1)
  sim._mjmodel.nhistory = 1
  with pytest.raises(ValueError, match="stepFD does not support actuator delays"):
    fd.mjd_stepFD(sim)
  with pytest.raises(ValueError, match="transitionFD does not support actuator delays"):
    fd.mjd_transitionFD(sim)
  sim._mjmodel.nhistory = 0
  sim._mjmodel.opt.integrator = int(mujoco.mjtIntegrator.mjINT_RK4)
  with pytest.raises(ValueError, match="transitionFD does not support RK4"):
    fd.mjd_transitionFD(sim)
  with pytest.raises(ValueError, match="inverseFD does not support RK4"):
    fd.mjd_inverseFD(sim)
  sim._mjmodel.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
  sim._mjmodel.opt.noslip_iterations = 2
  with pytest.raises(ValueError, match="inverseFD does not support the noslip solver"):
    fd.mjd_inverseFD(sim)
  sim._mjmodel.opt.noslip_iterations = 0
  sim._mjmodel.opt.enableflags = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  assert fd.mjd_transitionFD(sim)["A"].shape == (1, 3, 3)
  assert fd.mjd_inverseFD(sim)["DfDq"].shape == (1, 1, 1)
  assert cpu_query


def test_transition_and_inverse_fd_match_independent_fake_differences(cpu_query):
  sim = _Sim(batch=2)
  transition = fd.mjd_transitionFD(sim, eps=1e-3, flg_centered=False)
  assert transition["A"].shape == (2, 3, 3)
  assert transition["B"].shape == (2, 3, 1)
  assert transition["C"].shape == (2, 1, 3)
  assert transition["D"].shape == (2, 1, 1)
  torch.testing.assert_close(transition["B"][:, 1, 0], torch.full((2,), .75),
                             atol=2e-4, rtol=0)

  result = fd.mjd_inverseFD(sim, eps=1e-3)
  torch.testing.assert_close(result["DfDq"][:, 0, 0], torch.full((2,), 3.3),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DfDv"][:, 0, 0], torch.full((2,), 4.),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DfDa"][:, 0, 0], torch.full((2,), 2.2),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DsDq"][:, 0, 0], torch.full((2,), 1.),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DsDv"][:, 0, 0], torch.full((2,), 2.),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DsDa"][:, 0, 0], torch.full((2,), 3.),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DmDq"][:, 0, 0], torch.full((2,), 1.),
                             atol=3e-4, rtol=0)


class _OptionalLowSensorOwner:
  __module__ = "mujoco_metal.fake_optional_low_sensor_owner"

  def __init__(self, has_low_attribute):
    if has_low_attribute:
      self._s_state_out_low = None


def test_inverse_fd_low_seed_does_not_read_tensor_values_on_host(monkeypatch):
  def forbidden(*_args, **_kwargs):
    raise AssertionError("sensor low seed may not read device tensor contents")

  monkeypatch.setattr(torch, "count_nonzero", forbidden)
  for value in (0.0, 0.125):
    producer = _OptionalLowSensorOwner(False)
    sim = SimpleNamespace(_sensors=producer, _raw_sensor_program=None)
    seed = torch.full((1, 2), value, dtype=torch.float32)
    fd._inverse_fd_seed_sensor_low(sim, seed)
    assert producer._s_state_out_low is seed


@pytest.mark.parametrize("has_low_attribute", [False, True])
def test_inverse_fd_treats_missing_or_none_sensor_low_as_zero(
    cpu_query, has_low_attribute):
  """A sensor program may exist before allocating its optional low output."""
  sim = _Sim(batch=2)
  sensor_owner = _OptionalLowSensorOwner(has_low_attribute)
  sim._sensors = sensor_owner
  before = sim._sensordata.clone()
  result = fd.mjd_inverseFD(sim, eps=1e-3)
  torch.testing.assert_close(result["DsDq"][:, 0, 0], torch.ones(2),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DsDv"][:, 0, 0], torch.full((2,), 2.),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(sim._sensordata, before)
  if has_low_attribute:
    assert sensor_owner._s_state_out_low is None


class _RawSensorOwner:
  __module__ = "mujoco_metal.fake_raw_sensor_owner"

  def __init__(self, batch):
    self._s_state_out_low = torch.zeros((batch, 1), dtype=torch.float32)


def test_inverse_fd_reads_low_plane_from_raw_sensor_program_and_rolls_it_back(
    cpu_query):
  sim = _Sim(batch=2)
  raw_program = _RawSensorOwner(sim.batch_size)
  sim._sensors = None
  sim._raw_sensor_program = raw_program
  inverse_skip = sim.inverse_skip

  def inverse_skip_with_low(*args, **kwargs):
    result = inverse_skip(*args, **kwargs)
    raw_program._s_state_out_low.copy_(.25 * sim._state._qpos)
    return result

  sim.inverse_skip = inverse_skip_with_low
  saved_low = raw_program._s_state_out_low.clone()
  result = fd.mjd_inverseFD(sim, eps=1e-3)
  torch.testing.assert_close(result["DsDq"][:, 0, 0], torch.full((2,), 1.25),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(raw_program._s_state_out_low, saved_low)


def test_inverse_fd_materializes_missing_sensor_before_pinned_stage_calls(cpu_query):
  """Pre-step None storage still follows inverseSkip's per-column stages."""
  sim = _Sim(batch=2)
  sim._sensordata = None
  sim._sensors = _OptionalLowSensorOwner(True)
  sim._raw_sensor_program = sim._sensors
  calls = []
  prefix_sensor_flags = []
  original_inverse_skip = sim.inverse_skip
  original_pos = sim.prepare_forward_position
  original_vel = sim.prepare_forward_velocity

  def record_pos(*args, **kwargs):
    prefix_sensor_flags.append(bool(kwargs.get("skipsensor", False)))
    return original_pos(*args, **kwargs)

  def record_vel(*args, **kwargs):
    prefix_sensor_flags.append(bool(kwargs.get("skipsensor", False)))
    return original_vel(*args, **kwargs)

  sim.prepare_forward_position = record_pos
  sim.prepare_forward_velocity = record_vel

  def record_stages(*args, **kwargs):
    calls.append((kwargs.get("skipstage", mujoco.mjtStage.mjSTAGE_NONE),
                  kwargs.get("sensor_qacc_low"), kwargs.get("record")))
    return original_inverse_skip(*args, **kwargs)

  sim.inverse_skip = record_stages
  before_state = (sim._state._qpos.clone(), sim._state._qvel.clone(),
                  sim._state._qacc.clone())
  result = fd.mjd_inverseFD(sim, eps=1e-3)
  torch.testing.assert_close(result["DsDq"][:, 0, 0], torch.ones(2),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DsDv"][:, 0, 0], torch.full((2,), 2.),
                             atol=3e-4, rtol=0)
  torch.testing.assert_close(result["DsDa"][:, 0, 0], torch.full((2,), 3.),
                             atol=3e-4, rtol=0)
  assert [int(stage) for stage, _, _ in calls] == [
      int(mujoco.mjtStage.mjSTAGE_NONE),
      int(mujoco.mjtStage.mjSTAGE_VEL),
      int(mujoco.mjtStage.mjSTAGE_POS),
      int(mujoco.mjtStage.mjSTAGE_NONE)]
  assert prefix_sensor_flags and all(prefix_sensor_flags)
  # The VEL producer returns its values mapping, while the coordinator-owned
  # POS record remains the object inverseSkip validates and consumes.
  prepared = calls[1][2]
  assert prepared is not None
  assert ForwardStage.POS in prepared.values
  assert ForwardStage.VEL in prepared.values
  assert sim._sensordata is None
  torch.testing.assert_close(sim._state._qpos, before_state[0])
  torch.testing.assert_close(sim._state._qvel, before_state[1])
  torch.testing.assert_close(sim._state._qacc, before_state[2])


def test_inverse_fd_disabled_sensors_skip_prepared_prefix_producers(cpu_query):
  sim = _Sim(batch=1)
  sim._mjmodel.opt.disableflags = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  sim._sensordata.fill_(17.)
  sim._raw_sensordata = None
  disabled_low = torch.full_like(sim._sensordata, .25)
  sim._sensors = SimpleNamespace(_s_state_out_low=disabled_low)
  qacc_low_calls = []
  sim._qacc_low_for_sensor = lambda qacc: qacc_low_calls.append(qacc)
  skip_calls = []
  prefix_calls = []
  original = sim.inverse_skip
  original_pos = sim.prepare_forward_position
  original_vel = sim.prepare_forward_velocity

  def observe(*args, **kwargs):
    skip_calls.append(bool(kwargs.get("skipsensor", False)))
    return original(*args, **kwargs)

  def observe_pos(*args, **kwargs):
    prefix_calls.append(bool(kwargs.get("skipsensor", False)))
    return original_pos(*args, **kwargs)

  def observe_vel(*args, **kwargs):
    prefix_calls.append(bool(kwargs.get("skipsensor", False)))
    return original_vel(*args, **kwargs)

  sim.inverse_skip = observe
  sim.prepare_forward_position = observe_pos
  sim.prepare_forward_velocity = observe_vel
  result = fd.mjd_inverseFD(sim, eps=1e-3)
  assert skip_calls == [True] * 4
  assert prefix_calls == [True, True, True]
  # The supplied sample is retained when mjDSBL_SENSOR suppresses all
  # position, velocity, and acceleration sensor producers.
  torch.testing.assert_close(sim._sensordata, torch.full((1, 1), 17.))
  assert sim._raw_sensordata is None
  assert qacc_low_calls == []
  assert torch.count_nonzero(result["DsDq"]) == 0
  assert torch.count_nonzero(result["DsDv"]) == 0
  assert torch.count_nonzero(result["DsDa"]) == 0


def test_inverse_fd_preflights_lazy_sensor_backings_before_query(cpu_query):
  sim = _Sim(batch=2)
  sim._sensordata = None
  sim._raw_sensordata = None
  sim._sensors = SimpleNamespace(_s_state_out_low=None)
  sim._raw_sensor_program = sim._sensors
  before = (sim._state._qpos.clone(), sim._state._qvel.clone(),
            sim._state._qacc.clone(), sim._control.clone())
  sim.limits.memory_budget_bytes = 1

  with pytest.raises(ValueError, match="memory budget"):
    fd.mjd_inverseFD(sim, eps=1e-3)

  assert cpu_query == []
  assert sim._sensordata is None
  assert sim._raw_sensordata is None
  for actual, expected in zip((sim._state._qpos, sim._state._qvel,
                              sim._state._qacc, sim._control), before):
    torch.testing.assert_close(actual, expected)


def test_inverse_fd_preflight_charges_live_pair_rows_before_query(cpu_query,
                                                                  monkeypatch):
  baseline = _Sim(batch=2)
  paired = _Sim(batch=2)
  paired._coupled_constraints = SimpleNamespace(
      descriptor=SimpleNamespace(nr=5))
  requested = []
  monkeypatch.setattr(fd, "_preflight",
                      lambda _sim, size: requested.append(int(size)))

  fd.mjd_inverseFD(baseline, eps=1e-3)
  fd.mjd_inverseFD(paired, eps=1e-3)

  # The paired reducer owns six transient low/product row planes per world;
  # the extra two acceleration-sized planes cover baseline/candidate retained
  # constraint low words and remain charged even when the row span is empty.
  assert requested[1] - requested[0] == 2 * 6 * 5 * 4
  assert cpu_query == [baseline, paired]


def test_inverse_fd_sample_uses_device_state_tensor_not_public_device_string():
  model = mujoco.MjModel.from_xml_string("""<mujoco><option>
    <flag contact='disable'/></option><worldbody>
    <body><joint type='hinge'/><geom type='sphere' size='.1' mass='1'/></body>
  </worldbody></mujoco>""")
  state = DeviceState(model, validate_stepping_profile(model), 1, device="cpu")
  assert isinstance(state.device, str)
  assert isinstance(state._qvel.device, torch.device)
  sample = torch.zeros((1, model.nsensordata), dtype=torch.float32)
  sim = SimpleNamespace(_state=state, batch_size=1, device=state.device,
                        _sensordata=sample)
  borrowed = fd._inverse_fd_sensor_sample(sim, {}, model.nsensordata,
                                          copy=False)
  assert borrowed is sample


def test_inverse_fd_threads_provenance_matched_qacc_low_through_all_stages(cpu_query):
  sim = _Sim(batch=1)
  low = torch.full_like(sim._state._qacc, .125)
  sim._qacc_low_for_sensor = lambda qacc: low
  seen = []
  original = sim.inverse_skip

  def observe(*args, **kwargs):
    value = kwargs.get("sensor_qacc_low")
    seen.append(None if value is None else value.clone())
    return original(*args, **kwargs)

  sim.inverse_skip = observe
  fd.mjd_inverseFD(sim, eps=1e-3)
  assert len(seen) == 4
  assert all(value is not None for value in seen)
  for value in seen:
    torch.testing.assert_close(value, low)


def test_inverse_fd_carries_actuator_sensor_inputs_between_rolled_back_columns(
    cpu_query):
  class ActuatorSensorSim(_Sim):
    def __init__(self):
      super().__init__(batch=1, ns=1)
      self._has_acc_sensors = True
      self._sen_act_force = torch.zeros((1, 1), dtype=torch.float32)
      self._sen_qfrc_act = torch.zeros((1, 1), dtype=torch.float32)

    def inverse_skip(self, skipstage=mujoco.mjtStage.mjSTAGE_NONE,
                     skipsensor=False, *, return_components=False,
                     sensor_qacc_low=None, record=None):
      del skipstage, sensor_qacc_low
      self._inverse_records.append(record)
      q, v, a = self._state._qpos, self._state._qvel, self._state._qacc
      mass = (2 + q).reshape(self.batch_size, 1, 1)
      mass_qacc = mass[:, 0, 0, None] * a
      bias, passive = 3 * q, -4 * v
      constraint = torch.zeros_like(mass_qacc)
      force = bias + (mass_qacc - passive - constraint)
      if not skipsensor:
        # This models the ACC sensor's read of the prior ACT stage. Pinned
        # inverseSkip writes sensors first and runs fwdActuation afterwards.
        self._sensordata.copy_(self._sen_qfrc_act[:, :1])
      result = {
          "qfrc_inverse": force,
          "status": self._state._status.clone(),
          "record": SimpleNamespace(values={
              ForwardStage.VEL: {"dynamics": {"mass_matrix": mass}}}),
      }
      if return_components:
        result["qfrc_inverse_components"] = {
            "mass_qacc": mass_qacc, "bias": bias,
            "bias_low": torch.zeros_like(bias), "passive": passive,
            "constraint": constraint,
        }
      return result

    def prepare_forward_actuation(self, record):
      del record
      # The query has exactly one slide actuator force equal to its position.
      force = self._state._qpos.clone()
      self._sen_act_force.copy_(force)
      self._sen_qfrc_act.copy_(force)
      return {"qfrc_actuator": force}

  sim = ActuatorSensorSim()
  original_qfrc_act = sim._sen_qfrc_act.clone()
  result = fd.mjd_inverseFD(sim, eps=1e-3, flg_actuation=True)
  # Center sensors see entry ACT=0; each perturbed sensor sees the preceding
  # sample's ACT result, retained across tx.restore(). The qpos sample thus
  # produces the pinned sequential difference 0.2 / 1e-3.
  torch.testing.assert_close(result["DsDq"][:, 0, 0], torch.tensor([200.]),
                             atol=1e-4, rtol=0)
  torch.testing.assert_close(sim._sen_qfrc_act, original_qfrc_act)
  torch.testing.assert_close(sim._sen_act_force, torch.zeros_like(original_qfrc_act))


def test_inverse_fd_uses_pinned_multi_dof_acc_vel_pos_sample_order(cpu_query):
  class MultiDofActuatorSensorSim(_Sim):
    def __init__(self):
      super().__init__(batch=1, ns=1)
      model = self._mjmodel
      model.nq = model.nv = 2
      model.nu = 1
      model.na = 0
      model.nM = model.nC = 3
      model.dof_Madr = np.asarray([0, 1], np.int32)
      model.dof_parentid = np.asarray([-1, 0], np.int32)
      self._state = _State(1, nq=2, nv=2, na=0)
      self._state._qpos.copy_(torch.tensor([[.2, .4]]))
      self._state._qvel = torch.tensor([[.1, .3]])
      self._state._qacc.copy_(torch.tensor([[.05, .07]]))
      self._state._act = None
      self._control = torch.zeros((1, 1), dtype=torch.float32)
      self._has_acc_sensors = True
      self._sen_act_force = torch.zeros((1, 1), dtype=torch.float32)
      self._sen_qfrc_act = torch.zeros((1, 2), dtype=torch.float32)

    def inverse_skip(self, skipstage=mujoco.mjtStage.mjSTAGE_NONE,
                     skipsensor=False, *, return_components=False,
                     sensor_qacc_low=None, record=None):
      del skipstage, sensor_qacc_low
      self._inverse_records.append(record)
      q, v, a = self._state._qpos, self._state._qvel, self._state._qacc
      mass = torch.eye(2, dtype=torch.float32).reshape(1, 2, 2)
      if not skipsensor:
        self._sensordata.copy_(self._sen_qfrc_act[:, :1])
      mass_qacc = a.clone()
      bias, passive = 3 * q, -4 * v
      constraint = torch.zeros_like(mass_qacc)
      result = {
          "qfrc_inverse": bias + (mass_qacc - passive - constraint),
          "status": self._state._status.clone(),
          "record": SimpleNamespace(values={
              ForwardStage.VEL: {"dynamics": {"mass_matrix": mass}}}),
      }
      if return_components:
        result["qfrc_inverse_components"] = {
            "mass_qacc": mass_qacc, "bias": bias,
            "bias_low": torch.zeros_like(bias), "passive": passive,
            "constraint": constraint,
        }
      return result

    def prepare_forward_actuation(self, record):
      del record
      q, v, a = (self._state._qpos, self._state._qvel,
                 self._state._qacc)
      signal = (.13 * q[:, 0] + .17 * q[:, 1] +
                .19 * v[:, 0] + .23 * v[:, 1] +
                .29 * a[:, 0] + .31 * a[:, 1])
      force = torch.stack((signal, torch.zeros_like(signal)), dim=1)
      self._sen_act_force[:, 0].copy_(signal)
      self._sen_qfrc_act.copy_(force)
      return {"qfrc_actuator": force}

  sim = MultiDofActuatorSensorSim()
  initial = tuple(value.clone() for value in (
      sim._state._qpos, sim._state._qvel, sim._state._qacc))
  eps = 1e-3
  result = fd.mjd_inverseFD(sim, eps=eps, flg_actuation=True)

  # Independent execution of the pinned C loop: baseline sample first, then
  # each acceleration, velocity and position perturbation; sensor reads the
  # last ACT output and ACT updates the state carried into the next sample.
  q0, v0, a0 = (value.double().numpy()[0] for value in initial)
  def act_signal(q, v, a):
    return .13*q[0] + .17*q[1] + .19*v[0] + .23*v[1] + .29*a[0] + .31*a[1]
  sensor_center = 0.0
  cursor = act_signal(q0, v0, a0)
  expected = {"DsDa": np.zeros((2,)), "DsDv": np.zeros((2,)),
              "DsDq": np.zeros((2,))}
  for state, key, base in ((a0, "DsDa", a0), (v0, "DsDv", v0),
                           (q0, "DsDq", q0)):
    for i in range(2):
      q, v, a = q0.copy(), v0.copy(), a0.copy()
      changed = {"DsDa": a, "DsDv": v, "DsDq": q}[key]
      changed[i] += eps
      sample = cursor
      expected[key][i] = (sample - sensor_center) / eps
      cursor = act_signal(q, v, a)
  for key in expected:
    torch.testing.assert_close(
        result[key][0, :, 0], torch.as_tensor(expected[key], dtype=torch.float32),
        atol=2e-4, rtol=0)
  for actual, original in zip(
      (sim._state._qpos, sim._state._qvel, sim._state._qacc), initial):
    torch.testing.assert_close(actual, original)
  torch.testing.assert_close(sim._sen_qfrc_act,
                             torch.zeros_like(sim._sen_qfrc_act))


def test_inverse_fd_low_plane_follows_delayed_and_interval_history_selection():
  class History:
    entries = np.asarray([[1, 0, 4, 1, 0, 0],
                          [1, 4, 4, 1, 0, 1]], dtype=np.int32)

    def sensor_compute_mask(self, history, time):
      return torch.tensor([[False, False, True, True]], dtype=torch.bool)

  sim = _Sim(batch=1, ns=4)
  sim._mjmodel.sensor_adr = np.asarray([0, 1, 2], dtype=np.int32)
  sim._mjmodel.sensor_dim = np.asarray([1, 1, 2], dtype=np.int32)
  sim._mjmodel.sensor_delay = np.asarray([.1, 0., 0.], dtype=np.float64)
  sim._mjmodel.sensor_interval = np.asarray([[0., 0.], [.1, 0.], [0., 0.]])
  sim._state._history = torch.zeros((1, 1), dtype=torch.float32)
  sim._state._time = torch.zeros((1,), dtype=torch.float32)
  sim._history_program = History()
  owner = SimpleNamespace(_s_state_out_low=torch.tensor([[1., 2., 3., 4.]]))
  sim._raw_sensor_program = owner
  paired = fd._inverse_fd_sensor_low(sim, torch.zeros((1, 4)))
  torch.testing.assert_close(paired, torch.tensor([[0., 0., 3., 4.]]))
  torch.testing.assert_close(owner._s_state_out_low,
                             torch.tensor([[1., 2., 3., 4.]]))
