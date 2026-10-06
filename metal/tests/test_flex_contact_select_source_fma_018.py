"""CPU witness for the pinned arm64 farthest-point contact filter."""
from __future__ import annotations

import ctypes
import ctypes.util

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import lower_flex_contacts
from test_flex_fps_input_contract_018 import _fixture, _identities


def _fma(a: float, b: float, c: float) -> float:
  library = ctypes.CDLL(ctypes.util.find_library("m") or None)
  function = library.fma
  function.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
  function.restype = ctypes.c_double
  return float(function(float(a), float(b), float(c)))


def _source_filter_score(delta: np.ndarray) -> float:
  """Match the pinned arm64 instruction order for dx*dx+dy*dy+dz*dz."""
  x, y, z = map(float, delta)
  yy = y * y
  xy = _fma(x, x, yy)
  return _fma(z, z, xy)


def _source_filter(descriptor, distance: np.ndarray, point: np.ndarray,
                   maxkeep: int = 50, score=_source_filter_score) -> set[int]:
  groups = np.asarray(descriptor.filter_group, np.int32)
  selected_slots: set[int] = set()
  for group in np.unique(groups):
    slots = [int(i) for i in np.flatnonzero(groups == group)]
    if len(slots) <= maxkeep:
      selected_slots.update(slots)
      continue
    # filterFlexContacts stores selected/min_dist by mutable array position.
    permutation = slots.copy()
    selected = [False] * len(slots)
    min_distance = [1.0e10] * len(slots)
    best = 0
    best_depth = -float(distance[permutation[0]])
    for i in range(1, len(permutation)):
      depth = -float(distance[permutation[i]])
      if depth > best_depth:
        best, best_depth = i, depth
    nselected = 0
    while nselected < maxkeep and best >= 0:
      selected[best] = True
      best_position = point[permutation[best]]
      next_best = -1
      next_best_distance = -1.0
      for i, slot in enumerate(permutation):
        if selected[i]:
          continue
        d2 = score(point[slot] - best_position)
        if d2 < min_distance[i]:
          min_distance[i] = d2
        if min_distance[i] > next_best_distance:
          next_best_distance = min_distance[i]
          next_best = i
      if nselected < maxkeep - 1:
        permutation[nselected], permutation[best] = (
            permutation[best], permutation[nselected])
        if next_best == nselected:
          next_best = best
      nselected += 1
      best = next_best
    selected_slots.update(permutation[:nselected])
  return selected_slots


@pytest.mark.parametrize("midphase", [False, True])
def test_pinned_flex_fps_arm64_fma_replay_matches_full_cpu_contacts(midphase):
  model, reference = _fixture(midphase)
  data = mujoco.MjData(model)
  data.qpos[:] = reference.qpos
  data.qvel[:] = reference.qvel
  mujoco.mj_forward(model, data)
  vertices = np.asarray(data.flexvert_xpos, np.float64).reshape(-1, 3)
  descriptor = lower_flex_contacts(model)
  geom = np.asarray(descriptor.geom, np.int32)
  vert = np.asarray(descriptor.vert1, np.int32)
  distance = np.full(descriptor.slot_count, np.inf, np.float64)
  point = np.zeros((descriptor.slot_count, 3), np.float64)
  for slot, (geom_id, vert_id) in enumerate(zip(geom, vert)):
    if geom_id < 0 or vert_id < 0:
      continue
    normal = np.asarray(data.geom_xmat[geom_id], np.float64).reshape(3, 3)[2]
    delta = vertices[vert_id] - data.geom_xpos[geom_id]
    plane_distance = float(np.dot(delta, normal))
    radius = float(model.flex_radius[int(descriptor.flex1[slot])])
    distance[slot] = plane_distance - radius
    point[slot] = vertices[vert_id] + normal * (
        -distance[slot] * 0.5 - radius)

  # The independent pinned CPU collision call remains the identity oracle.
  mujoco.mj_collision(model, data)
  expected = _identities(data)
  selected = _source_filter(descriptor, distance, point)
  replayed = {(int(geom[slot]), int(vert[slot])) for slot in selected}
  assert replayed == expected
  assert len(replayed) == len(expected) == (50 if midphase else 100)

  # Narrowing source contact midpoints to a float high word loses ties that
  # the pinned 53-bit FMA sequence resolves. This guards the low/tail selector
  # ABI requirement independently of the production MSL gate.
  point_high = point.astype(np.float32).astype(np.float64)
  high_selected = _source_filter(descriptor, distance, point_high)
  high_ids = {(int(geom[slot]), int(vert[slot])) for slot in high_selected}
  assert high_ids != expected
