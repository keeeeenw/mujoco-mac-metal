# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Residual words are state, not a disposable position-stage cache."""
from dataclasses import replace
import numpy as np
import pytest
from test_device_state import _state
from mujoco_metal.retained_state import pack_residuals


def _represented(state):
  w=state._precision.synchronize(state._qpos,state._qvel).cpu().numpy()
  nq=state._model.nq;nv=state._model.nv
  return (state.qpos.cpu().numpy().astype(float)+w[:,:nq]+w[:,nq:2*nq],
          state.qvel.cpu().numpy().astype(float)+w[:,2*nq:2*nq+nv]+w[:,2*nq+nv:])


def test_host_inputs_checkpoint_selected_restore_copy_and_legacy():
  q=np.array([[.123456789012345],[.234567890123456],[.345678901234567]])
  v=-q*.3141592653589793
  s=_state(qpos=q,qvel=v)
  for actual,expected in zip(_represented(s),(q,v)):np.testing.assert_array_equal(actual,expected)
  snap=s.snapshot();assert snap.schema_version==7 and not snap.precision_words.flags.writeable
  s.reset([1],qpos=[.876543210987654],qvel=[.987654321098765])
  np.testing.assert_array_equal(_represented(s)[0][[0,2]],q[[0,2]])
  s.restore(snap,env_ids=[1]);np.testing.assert_array_equal(_represented(s)[0],q)
  s.copy_environment(0,2)
  np.testing.assert_array_equal(_represented(s)[0][2],q[0])
  s.restore(replace(snap,schema_version=6))
  np.testing.assert_array_equal(s._precision.words.numpy(),0)
  np.testing.assert_array_equal(_represented(s)[0],snap.qpos)


def test_direct_mutation_replacement_unavailable_counters_and_masked_reset():
  import torch
  q=np.full((3,1),.123456789012345);v=np.full((3,1),.234567890123456)
  s=_state(qpos=q,qvel=v)
  original=s.snapshot().precision_words.copy()
  s._qpos[1,0]+=.25
  s._precision.synchronize(s._qpos,s._qvel)
  np.testing.assert_array_equal(s._precision.words.numpy()[[0,2]],original[[0,2]])
  np.testing.assert_array_equal(s._precision.words.numpy()[1,:2],0)
  np.testing.assert_array_equal(s._precision.words.numpy()[1,2:],original[1,2:])
  # No mutation counter is consulted: direct reconciliation still detects
  # real bit changes, including writes made by an external native producer.
  s._qvel[2,0]=-.25
  s._precision.synchronize(s._qpos,s._qvel)
  np.testing.assert_array_equal(s._precision.words.numpy()[2,2:],0)
  s._qpos=s._qpos.clone()
  s._precision.synchronize(s._qpos,s._qvel)
  np.testing.assert_array_equal(s._precision.words.numpy()[:,:2],0)
  s.restore(s.snapshot())
  before=s._precision.words.clone()
  s.reset_mask(torch.tensor([False,True,False]))
  np.testing.assert_array_equal(s._precision.words.numpy()[[0,2]],before.numpy()[[0,2]])
  np.testing.assert_array_equal(s._precision.words.numpy()[1],0)


def test_restore_precision_preflight_is_atomic_and_probe_is_detached():
  q=np.full((3,1),.123456789012345);s=_state(qpos=q,qvel=-q)
  good=s.snapshot();bad=replace(good)
  words=good.precision_words.copy();words[0,0]=float('nan')
  object.__setattr__(bad,'precision_words',words)
  with pytest.raises(ValueError,match='finite'):s.restore(bad)
  np.testing.assert_array_equal(s.snapshot().precision_words,good.precision_words)
  s.validate_snapshot(good)
  np.testing.assert_array_equal(s.snapshot().precision_words,good.precision_words)
  words=good.precision_words.copy();words[0,0]=.5
  with pytest.raises(ValueError,match='projection'):replace(good,precision_words=words)


def test_unrepresentable_source_words_fail_before_mutating_any_live_state():
  s=_state(qpos=np.full((3,1),.123456789012345))
  good=s.snapshot()
  with pytest.raises(ValueError,match='exact hi\\+lo\\+tail'):
    s.reset([1],qpos=[1e-100])
  np.testing.assert_array_equal(s.snapshot().qpos,good.qpos)
  np.testing.assert_array_equal(s.snapshot().precision_words,good.precision_words)
  with pytest.raises(ValueError,match='exact hi\\+lo\\+tail'):
    _state(qvel=np.full((3,1),1e-100))


def test_signed_zero_high_projections_round_trip_without_changing_state_bits():
  from mujoco_metal.retained_state import validate_residuals
  q = np.array([[-0.0], [0.0], [-0.0]], dtype=np.float32)
  v = np.array([[0.0], [-0.0], [-0.0]], dtype=np.float32)
  words = pack_residuals(q, v)
  accepted, represented_q, represented_v = validate_residuals(q, v, words)
  np.testing.assert_array_equal(accepted.view(np.uint32), words.view(np.uint32))
  np.testing.assert_array_equal(represented_q.astype(np.float32).view(np.uint32), q.view(np.uint32))
  np.testing.assert_array_equal(represented_v.astype(np.float32).view(np.uint32), v.view(np.uint32))
  state = _state(qpos=q, qvel=v)
  saved = state.snapshot()
  state.reset([1], qpos=[.1], qvel=[.2])
  state.restore(saved)
  np.testing.assert_array_equal(state.snapshot().qpos.view(np.uint32), q.view(np.uint32))
  np.testing.assert_array_equal(state.snapshot().qvel.view(np.uint32), v.view(np.uint32))


def test_signed_zero_admission_still_rejects_nonzero_projection_and_noncanonical_words():
  from mujoco_metal.retained_state import validate_residuals
  q = np.array([[-0.0]], dtype=np.float32)
  v = np.zeros_like(q)
  words = pack_residuals(q, v)
  words[0, 0] = .125
  with pytest.raises(ValueError, match="projection"):
    validate_residuals(q, v, words)
  words.fill(0)
  words[0, 1] = -.0
  with pytest.raises(ValueError, match="canonical"):
    validate_residuals(q, v, words)
