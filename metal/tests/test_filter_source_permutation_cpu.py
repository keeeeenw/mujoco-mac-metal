"""CPU replay for the position-indexed filterFlexContacts port.

The native test in test_flex_contact_lowering.py retains the original
installed-CPU identity oracle. These tests isolate the mutable contact-array
state and the strict binary64 score ordering without requiring MPS.
"""

from pathlib import Path
import re

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _pinned_positions(points, distances, maxkeep=50):
  """Return source-selected IDs and the final in-place contact permutation."""
  points = np.asarray(points, dtype=np.float64)
  distances = np.asarray(distances, dtype=np.float64)
  count = len(points)
  if count <= maxkeep:
    return list(range(count)), list(range(count))
  selected = np.zeros(count, dtype=np.uint8)
  min_dist = np.full(count, 1.0e10, dtype=np.float64)  # pinned mjMAXVAL
  permutation = list(range(count))
  best = 0
  bestdist = -distances[0]
  for i in range(1, count):
    candidate = -distances[i]
    if candidate > bestdist:
      bestdist, best = candidate, i
  nselected = 0
  while nselected < maxkeep and best >= 0:
    selected[best] = 1
    bestpos = points[permutation[best]]
    nextbest = -1
    nextbestdist = -1.0
    for i in range(count):
      if selected[i]:
        continue
      delta = points[permutation[i]] - bestpos
      d2 = delta[0] * delta[0] + delta[1] * delta[1]
      d2 = d2 + delta[2] * delta[2]
      if d2 < min_dist[i]:
        min_dist[i] = d2
      if min_dist[i] > nextbestdist:
        nextbestdist, nextbest = min_dist[i], i
    if nselected < maxkeep - 1:
      permutation[nselected], permutation[best] = (
          permutation[best], permutation[nselected])
      if nextbest == nselected:
        nextbest = best
    nselected += 1
    best = nextbest
  return permutation[:nselected], permutation


def _shader_state(points, distances, maxkeep=50, dtype=np.float64):
  """Replay the MSL position-indexed state and fixed-slot score arithmetic."""
  points = np.asarray(points, dtype=dtype)
  distances = np.asarray(distances, dtype=dtype)
  count = len(points)
  permutation = list(range(count))
  selected = np.zeros(count, dtype=np.uint8)
  min_dist = np.full(count, dtype(1.0e10), dtype=dtype)
  best = 0
  bestdist = dtype(-distances[0])
  for i in range(1, count):
    candidate = dtype(-distances[i])
    if candidate > bestdist:
      bestdist, best = candidate, i
  nselected = 0
  while nselected < maxkeep and best >= 0:
    selected[best] = 1
    bestpos = points[permutation[best]]
    nextbest, nextbestdist = -1, dtype(-1.0)
    for i in range(count):
      if selected[i]:
        continue
      delta = np.asarray(points[permutation[i]] - bestpos, dtype=dtype)
      d2 = dtype(dtype(delta[0] * delta[0] + delta[1] * delta[1])
                 + dtype(delta[2] * delta[2]))
      if d2 < min_dist[i]:
        min_dist[i] = d2
      if min_dist[i] > nextbestdist:
        nextbestdist, nextbest = min_dist[i], i
    if nselected < maxkeep - 1:
      permutation[nselected], permutation[best] = (
          permutation[best], permutation[nselected])
      if nextbest == nselected:
        nextbest = best
    nselected += 1
    best = nextbest
  return permutation[:nselected]


def _shader_batched_state(points, distances, active, groups, world_mask,
                          maxkeep=3, dtype=np.float64):
  """Execute selector state with the production [B,nslot] scratch offsets."""
  points=np.asarray(points,dtype=np.float64)
  distances=np.asarray(distances,dtype=np.float64)
  active=np.asarray(active,dtype=bool)
  groups=np.asarray(groups,dtype=np.int32)
  world_mask=np.asarray(world_mask,dtype=np.int32)
  batch,nslot=active.shape
  out=np.full((batch,nslot),9,dtype=np.int32)
  permutation=np.full((batch,nslot),-1,dtype=np.int32)
  selected=np.zeros((batch,nslot),dtype=np.uint8)
  min_dist=np.empty((batch,nslot),dtype=dtype)
  for world in range(batch):
    if world_mask[world]==0:
      continue  # production early return preserves healthy outputs
    out[world].fill(0)
    for group in sorted(set(groups[active[world]])):
      slots=np.flatnonzero(active[world] & (groups==group))
      count=len(slots)
      if count<=maxkeep:
        out[world,slots]=1
        continue
      base=world*nslot
      selected[world].fill(0)
      min_dist[world].fill(1.0e10)
      permutation[world,:count]=slots
      best=0; bestdist=dtype(-distances[world,permutation[world,0]])
      for i in range(1,count):
        value=dtype(-distances[world,permutation[world,i]])
        if value>bestdist: bestdist,best=value,i
      nselected=0
      while nselected<maxkeep and best>=0:
        selected[world,best]=1
        bestpos=points[world,permutation[world,best]]
        nextbest=-1; nextbestdist=dtype(-1.0)
        for i in range(count):
          if selected[world,i]: continue
          delta=np.asarray(
              points[world,permutation[world,i]]-bestpos,dtype=dtype)
          d2=dtype(dtype(delta[0]*delta[0]+delta[1]*delta[1])
                   +dtype(delta[2]*delta[2]))
          if d2<min_dist[world,i]: min_dist[world,i]=d2
          if min_dist[world,i]>nextbestdist:
            nextbestdist,nextbest=min_dist[world,i],i
        if nselected<maxkeep-1:
          permutation[world,nselected],permutation[world,best]=(
              permutation[world,best],permutation[world,nselected])
          if nextbest==nselected: nextbest=best
        nselected+=1;best=nextbest
      out[world,permutation[world,:nselected]]=1
  return out


def test_source_position_state_matches_literal_permutation_and_preserves_ids():
  # Strict ties and a swap of the deepest contact exercise the non-permuted
  # selected/min_dist arrays, which an immutable selected-ID set cannot model.
  points = np.asarray([
      [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
      [2.0, 2.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0],
  ], dtype=np.float64)
  distances = np.asarray([0.5, 0.1, 0.2, 0.3, -0.2, 0.0], np.float64)
  expected, final_permutation = _pinned_positions(points, distances, 3)
  actual = _shader_state(points, distances, 3, np.float64)
  assert actual == expected
  assert len(set(actual)) == 3
  assert sorted(final_permutation) == list(range(len(points)))


def test_strict_score_rounding_changes_the_pinned_plane_fixture_set():
  # Values are distinct binary64 squared distances, but the low term is lost
  # when the source expression is evaluated in binary32. The third source
  # contact remains in the output prefix after the mutable swap, so this
  # strict tie changes the actual fixed-slot identity set.
  delta = np.float64(2.0 ** -12)
  points = np.asarray([
      [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
      [2.0, delta, 0.0], [0.0, 1.0, 0.0],
  ], dtype=np.float64)
  distances = np.zeros(4, dtype=np.float64)
  source_ids, _ = _pinned_positions(points, distances, 3)
  double_ids = _shader_state(points, distances, 3, np.float64)
  float_ids = _shader_state(points, distances, 3, np.float32)
  assert double_ids == source_ids == [0, 2, 1]
  assert float_ids == [0, 1, 2]


def test_candidate_binds_per_world_position_state_and_three_word_scores():
  host = (ROOT / "mujoco_metal/flex_contact.py").read_text()
  shader = (ROOT / "mujoco_metal/shaders/flex_contact.metal").read_text()
  assert "self._filter_permutation" in host
  assert "self._filter_selected_position" in host
  assert "self._filter_min_distance_words" in host
  assert "permutation [[buffer(13)]]" in shader
  assert "selected_position [[buffer(14)]]" in shader
  assert "min_distance_words [[buffer(15)]]" in shader
  assert "if (nextbest==nselected) nextbest=best;" in shader
  assert "min_distance_words[score_offset+2]=d2.tail;" in shader


def test_batched_selector_uses_world_local_state_and_preserves_healthy_world():
  points=np.asarray([
      [[0,0,0],[3,0,0],[0,3,0],[2,2,0],[1,0,0],[0,1,0],
       [2,0,0],[0,2,0],[1,1,0],[4,0,0]],
      [[2,2,0],[0,0,0],[1,1,0],[4,0,0],[0,4,0],[3,3,0],
       [2,0,0],[0,2,0],[5,0,0],[0,5,0]],
      [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[2,2,0],[3,3,0],
       [4,4,0],[5,5,0],[6,6,0],[7,7,0]],
  ],dtype=np.float64)
  distances=np.asarray([
      [0,-.2,-.1,-.3,0,0,0,0,0,0],
      [0,.2,0,-.1,0,0,-.3,0,0,0],
      np.zeros(10),
  ],dtype=np.float64)
  active=np.asarray([
      [0,0,1,0,1,0,0,1,1,0],   # group 0 exact cap (3), group 1 exact
      [1,1,0,1,0,1,1,0,0,0],   # world-local above-cap group 0 (4)
      [1,1,1,1,1,1,1,1,1,1],   # healthy world must be untouched
  ],dtype=bool)
  groups=np.asarray([0,0,0,1,0,1,0,0,1,0],dtype=np.int32)
  mask=np.asarray([1,1,0],dtype=np.int32)
  result=_shader_batched_state(points,distances,active,groups,mask,3)
  assert np.array_equal(result[0],active[0].astype(np.int32))
  expected=[]
  slots=np.flatnonzero(active[1] & (groups==0))
  expected.extend(slots[index] for index in _pinned_positions(
      points[1,slots],distances[1,slots],3)[0])
  expected.extend(np.flatnonzero(active[1] & (groups==1)))
  assert set(np.flatnonzero(result[1]))==set(expected)
  assert np.all(result[2]==9)
  # The production deepest-contact accesses must use the per-world base for
  # both the permutation entry and its corresponding distance.
  shader=(ROOT/"mujoco_metal/shaders/flex_contact.metal").read_text()
  assert "permutation[perm_base]]" in shader
  assert "permutation[perm_base+i]" in shader
