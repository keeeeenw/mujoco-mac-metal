# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Execute the public constant-cache recomputation contract against MuJoCo."""

from copy import copy

import mujoco
import numpy as np
import pytest

from mujoco_metal.lifecycle import BatchedConstants


XML = '''<mujoco><worldbody><body><freejoint/>
  <geom size=".1" mass="2"/><body pos="0 0 .3"><joint/>
    <geom size=".08" mass=".7"/></body></body></worldbody></mujoco>'''


def _oracle(model, masses):
  values = []
  for row in masses:
    candidate = copy(model)
    candidate.body_mass[:] = row
    mujoco.mj_setConst(candidate, mujoco.MjData(candidate))
    values.append(candidate.body_invweight0.copy())
  return np.asarray(values)


def _capture(cache):
  return (cache.body_mass.copy(), cache.body_invweight0.copy(),
          cache.row_generation.copy())


def _unchanged(cache, before):
  for actual, wanted in zip(_capture(cache), before):
    np.testing.assert_array_equal(actual, wanted)


def test_public_recompute_initial_unchanged_changed_and_explicit_world_rows(monkeypatch):
  source = mujoco.MjModel.from_xml_string(XML)
  cache = BatchedConstants(source, batch_size=3)
  assert cache.body_invweight0.shape == (3, *source.body_invweight0.shape)
  np.testing.assert_array_equal(cache.body_invweight0, _oracle(source, cache.body_mass))
  original = mujoco.mj_setConst
  calls = []
  def counted(model, data):
    calls.append(model.body_mass.copy())
    original(model, data)
  monkeypatch.setattr(mujoco, "mj_setConst", counted)
  candidates = cache.body_mass.copy()
  candidates[:, 1:] *= [[1.2], [.8], [1.7]]
  assert cache.recompute_constants({"body_mass": candidates}, [0, 1, 2]) == (0, 1, 2)
  assert len(calls) == 3
  np.testing.assert_array_equal(cache.row_generation, [1, 1, 1])
  calls.clear()
  assert cache.recompute_constants({"body_mass": candidates[[2, 0]]}, [2, 0]) == ()
  assert calls == []
  changed = candidates[[2, 0]].copy()
  changed[1, 2] *= 1.3
  assert cache.recompute_constants({"body_mass": changed}, [2, 0]) == (0,)
  assert len(calls) == 1
  wanted = candidates.copy(); wanted[0] = changed[1]
  np.testing.assert_array_equal(cache.body_mass, wanted)
  np.testing.assert_array_equal(cache.row_generation, [2, 1, 1])
  monkeypatch.setattr(mujoco, "mj_setConst", original)
  np.testing.assert_array_equal(cache.body_invweight0, _oracle(source, wanted))


@pytest.mark.parametrize("operation", ["recompute", "restore"])
def test_late_actual_recomputation_failure_keeps_all_rows_and_generations(operation, monkeypatch):
  source = mujoco.MjModel.from_xml_string(XML)
  cache = BatchedConstants(source, batch_size=3)
  snapshot = cache.snapshot()
  changed = cache.body_mass.copy(); changed[:, 1:] *= 1.4
  cache.recompute_constants({"body_mass": changed}, [0, 1, 2])
  before = _capture(cache)
  original = mujoco.mj_setConst
  calls = []
  def fail_second(model, data):
    original(model, data)
    calls.append(True)
    if len(calls) == 2:
      raise RuntimeError("late constant recomputation failure")
  monkeypatch.setattr(mujoco, "mj_setConst", fail_second)
  with pytest.raises(RuntimeError, match="late constant"):
    if operation == "restore":
      cache.restore(snapshot)
    else:
      changed[:, 1:] *= 1.2
      cache.recompute_constants({"body_mass": changed}, [2, 0, 1])
  assert len(calls) == 2
  _unchanged(cache, before)
  monkeypatch.setattr(mujoco, "mj_setConst", original)
  cache.restore(snapshot)
  np.testing.assert_array_equal(cache.body_invweight0, _oracle(source, snapshot.body_mass))
  # A retry after the failure must not incorrectly treat candidate masses as
  # committed cache keys, including when the rejected operation was restore.
  assert cache.recompute_constants({"body_mass": changed}, [0, 1, 2]) == (0, 1, 2)
  np.testing.assert_array_equal(cache.body_invweight0, _oracle(source, changed))


def test_restore_then_randomization_matches_fresh_replay_and_pinned_constants():
  source = mujoco.MjModel.from_xml_string(XML)
  cache = BatchedConstants(source, batch_size=3)
  checkpoint = cache.snapshot()
  cache.randomize([0, 1, 2], seed=12, scale=.2)
  cache.restore(checkpoint)
  cache.randomize([2, 0], seed=37, scale=.15)
  fresh = BatchedConstants(source, batch_size=3)
  fresh.restore(checkpoint)
  fresh.randomize([2, 0], seed=37, scale=.15)
  np.testing.assert_array_equal(cache.body_mass, fresh.body_mass)
  np.testing.assert_array_equal(cache.body_invweight0, fresh.body_invweight0)
  np.testing.assert_array_equal(cache.body_invweight0, _oracle(source, cache.body_mass))
  np.testing.assert_array_equal(cache.body_mass[1], checkpoint.body_mass[1])
