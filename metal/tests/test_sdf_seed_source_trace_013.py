# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded per-seed observer for the staged rigid SDF producer.

The GPU gate wraps the existing staged shader calls and clones their private
workspace after each dispatch.  The clones are read only after the normal
producer completes.  A scalar CPU replay follows the pinned 3.10
``mjc_SDF``/``stepGradient`` ordering for bundled-plugin SDF versus sphere
fixtures, so the first seed and stage that diverge is reported without
reconstructing descent points from rounded contacts.
"""

import ctypes
import hashlib
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from test_bundled_plugins_019 import _plugin_descriptor
from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
from mujoco_metal.simulation import MetalSimulation


_PLUGIN_NAMES = ("bowl", "torus")
_STAGE = {
    "x": 0,
    "x0": 9,
    "grad": 18,
    "dist0": 27,
    "alpha": 33,
    "wolfe": 36,
    "dist": 39,
    "t21": 42,
    "normal": 150,
}
_STATE_WORDS = 159


def _matvec(matrix, vector):
  result = np.zeros(3, dtype=np.float64)
  mujoco.mju_mulMatVec3(result, np.asarray(matrix, dtype=np.float64),
                        np.asarray(vector, dtype=np.float64))
  return result


def _mat_t_vec(matrix, vector):
  result = np.zeros(3, dtype=np.float64)
  mujoco.mju_mulMatTVec3(result, np.asarray(matrix, dtype=np.float64),
                         np.asarray(vector, dtype=np.float64))
  return result


def _geom_pos(data, geom):
  return np.asarray(data.geom_xpos, dtype=np.float64).reshape(-1)[
      3 * geom:3 * geom + 3]


def _geom_mat(data, geom):
  return np.asarray(data.geom_xmat, dtype=np.float64).reshape(-1)[
      9 * geom:9 * geom + 9]


def _pose_map(data, source_geom, target_geom):
  """Return pinned mapPose(source world pose, target world pose)."""
  source_quat = np.zeros(4, dtype=np.float64)
  target_quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(source_quat, _geom_mat(data, source_geom))
  mujoco.mju_mat2Quat(target_quat, _geom_mat(data, target_geom))
  neg_pos = np.zeros(3, dtype=np.float64)
  neg_quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_negPose(neg_pos, neg_quat, _geom_pos(data, target_geom),
                     target_quat)
  offset = np.zeros(3, dtype=np.float64)
  quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mulPose(offset, quat, neg_pos, neg_quat,
                     _geom_pos(data, source_geom), source_quat)
  rotation = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, quat)
  return offset, rotation


def _plugin_callbacks(model, data, geom_id):
  instance = int(model.geom_plugin[geom_id])
  plugin = _plugin_descriptor(model.plugin[instance])
  distance = ctypes.CFUNCTYPE(
      ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p,
      ctypes.c_int)(plugin.sdf_distance)
  gradient = ctypes.CFUNCTYPE(
      None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
      ctypes.c_void_p, ctypes.c_int)(plugin.sdf_gradient)
  address = ctypes.c_void_p(data._address)

  def sdf_distance(point):
    point = (ctypes.c_double * 3)(*np.asarray(point, dtype=np.float64))
    return float(distance(point, address, instance))

  def sdf_gradient(point):
    point = (ctypes.c_double * 3)(*np.asarray(point, dtype=np.float64))
    output = (ctypes.c_double * 3)()
    gradient(output, point, address, instance)
    return np.asarray(output, dtype=np.float64)

  return sdf_distance, sdf_gradient


def _source_function(source, signature):
  start = source.index(signature)
  brace = source.index("{", start)
  depth = 0
  for pos in range(brace, len(source)):
    if source[pos] == "{":
      depth += 1
    elif source[pos] == "}":
      depth -= 1
      if depth == 0:
        return source[start:pos + 1]
  raise AssertionError(f"unterminated source function: {signature}")


def _primitive_distance(model, geom, point):
  kind = int(model.geom_type[geom])
  size = np.asarray(model.geom_size, dtype=np.float64).reshape(-1)[
      3 * geom:3 * geom + 3]
  x, y, z = (float(v) for v in point)
  if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
    return float(mujoco.mju_norm3(np.asarray(point, dtype=np.float64)) - size[0])
  if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
    a = np.abs(point) - size
    if np.any(a >= 0):
      b = np.maximum(a, 0.0)
      return float(np.linalg.norm(b) + min(max(a[0], max(a[1], a[2])), 0.0))
    b = -size / a
    b /= np.linalg.norm(b)
    b = np.where(point < 0.0, -b, b)
    t = -a / np.abs(b)
    return float(-min(t[0], min(t[1], t[2])) * np.linalg.norm(b))
  raise AssertionError(f"source replay primitive unsupported: {kind}")


def _primitive_gradient(model, geom, point):
  kind = int(model.geom_type[geom])
  size = np.asarray(model.geom_size, dtype=np.float64).reshape(-1)[
      3 * geom:3 * geom + 3]
  if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
    result = np.asarray(point, dtype=np.float64).copy()
    mujoco.mju_normalize3(result)
    return result
  if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
    a = np.abs(point) - size
    k = 0 if a[0] > a[1] else 1
    l = 2 if a[2] > a[k] else k
    if a[l] < 0:
      b = -size / a
      b /= np.linalg.norm(b)
      return np.where(point < 0.0, -b, b)
    b = np.maximum(a, 0.0)
    c = np.linalg.norm(b)
    return np.array([b[i] / c * point[i] / abs(point[i]) if a[i] > 0 else 0.0
                     for i in range(3)], dtype=np.float64)
  raise AssertionError(f"source replay primitive unsupported: {kind}")


def _source_pair_replay(model, data, sdf_geom, analytic_geom):
  """Replay source AABB seeds and both source gradient-descent passes."""
  sdf_dist, sdf_grad = _plugin_callbacks(model, data, sdf_geom)
  offset21, rotation21 = _pose_map(data, sdf_geom, analytic_geom)
  offset12, rotation12 = _pose_map(data, analytic_geom, sdf_geom)
  geom_aabb = np.asarray(model.geom_aabb, dtype=np.float64).reshape(-1)
  aabb1 = geom_aabb[6 * analytic_geom:6 * analytic_geom + 6]
  aabb2 = geom_aabb[6 * sdf_geom:6 * sdf_geom + 6]
  lo1, hi1 = np.full(3, np.inf), np.full(3, -np.inf)
  lo2, hi2 = np.full(3, np.inf), np.full(3, -np.inf)
  for k in range(8):
    sign = np.array([1.0 if k & (1 << axis) else -1.0
                     for axis in range(3)])
    v1 = aabb1[:3] + aabb1[3:] * sign
    v2 = _matvec(rotation21, aabb2[:3] + aabb2[3:] * sign) + offset21
    lo1, hi1 = np.minimum(lo1, v1), np.maximum(hi1, v1)
    lo2, hi2 = np.minimum(lo2, v2), np.maximum(hi2, v2)
  lo, hi = np.maximum(lo1, lo2), np.minimum(hi1, hi2)

  def side_distance(point_sdf, point_analytic):
    return sdf_dist(point_sdf), _primitive_distance(model, analytic_geom,
                                                    point_analytic)

  def distance(mode, point):
    other = _matvec(rotation21, point) + offset21
    a, b = side_distance(point, other)
    if mode == 0:
      return max(a, b)
    if mode == 1:
      return a - b
    return a + b + abs(max(a, b))

  def gradient(mode, point):
    other = _matvec(rotation21, point) + offset21
    a, b = side_distance(point, other)
    ga = sdf_grad(point)
    gb = _mat_t_vec(rotation21, _primitive_gradient(model, analytic_geom,
                                                    other))
    if mode == 0:
      return ga if a > b else gb
    if mode == 1:
      mujoco.mju_normalize3(ga)
      mujoco.mju_normalize3(gb)
      result = np.zeros(3, dtype=np.float64)
      mujoco.mju_sub3(result, ga, gb)
      mujoco.mju_normalize3(result)
      return result
    picked = ga if a > b else gb
    result = np.zeros(3, dtype=np.float64)
    mujoco.mju_add3(result, ga, gb)
    mujoco.mju_addToScl3(result, picked, 1.0 if max(a, b) > 0.0 else -1.0)
    return result

  def descend(point, mode, niter):
    point = np.asarray(point, dtype=np.float64).copy()
    history = []
    for _ in range(niter):
      grad = gradient(mode, point)
      if any(np.isnan(grad[k]) or abs(grad[k]) > 1e10 for k in range(3)):
        return point, 1e10, history
      x0 = point.copy()
      dist0 = distance(mode, x0)
      alpha = 2.0
      wolfe = -0.1 * alpha * float(mujoco.mju_dot3(grad, grad))
      alpha0, wolfe0 = alpha, wolfe
      while True:
        alpha *= 0.5
        wolfe *= 0.5
        point = np.zeros(3, dtype=np.float64)
        mujoco.mju_addScl3(point, x0, grad, -alpha)
        dist = distance(mode, point)
        if not (alpha > 1e-4 and dist - dist0 > wolfe):
          break
      history.append({"x0": x0, "grad": grad.copy(), "dist0": dist0,
                      "alpha0": alpha0, "wolfe0": wolfe0,
                      "alpha": alpha, "wolfe": wolfe,
                      "x": point.copy(), "dist": dist})
      if dist0 < dist:
        # Pinned stepGradient returns the rejected trial distance and leaves
        # x at that trial point.  The next (intersection) pass starts there.
        return point, dist, history
    return point, dist, history

  accepted = []
  accepted_points = []
  seeds = []
  sdf_quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(sdf_quat, _geom_mat(data, sdf_geom))
  analytic_quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(analytic_quat, _geom_mat(data, analytic_geom))
  sdf_pos = _geom_pos(data, sdf_geom)
  for seed in range(int(model.opt.sdf_initpoints)):
    start = np.array([
        lo[0] + (hi[0] - lo[0]) * mujoco.mju_Halton(seed, 2),
        lo[1] + (hi[1] - lo[1]) * mujoco.mju_Halton(seed, 3),
        lo[2] + (hi[2] - lo[2]) * mujoco.mju_Halton(seed, 5)],
        dtype=np.float64)
    point = _matvec(rotation12, start) + offset12
    collision_x, collision_dist, collision = descend(
        point, 2, int(model.opt.sdf_iterations))
    intersection_x, intersection_dist, intersection = descend(
        collision_x, 0, 1)
    duplicate = (intersection_dist <= 0.0 and any(
        mujoco.mju_dist3(intersection_x, old) < 1e-15
        for old in accepted_points))
    contact = None
    normal = np.zeros(3, dtype=np.float64)
    if intersection_dist <= 0.0 and not duplicate:
      # addPreContact stores the candidate before checking normal degeneracy;
      # cnt advances only when the normalized normal is usable.
      normal = gradient(1, intersection_x)
      normal_length = mujoco.mju_normalize3(normal)
      if normal_length >= 1e-15:
        accepted_points.append(intersection_x.copy())
        normal = -normal
        world_normal = np.zeros(3, dtype=np.float64)
        world_point = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(world_normal, normal, sdf_quat)
        mujoco.mju_rotVecQuat(world_point, intersection_x, sdf_quat)
        world_point += sdf_pos - 0.5 * intersection_dist * world_normal
        contact = {"dist": intersection_dist, "normal": world_normal,
                   "pos": world_point}
        accepted.append((intersection_x.copy(), intersection_dist))
    seeds.append({"seed": seed, "x_init": point.copy(),
                  "collision_x": collision_x, "collision_dist": collision_dist,
                  "collision": collision, "intersection_x": intersection_x,
                  "intersection_dist": intersection_dist,
                  "intersection": intersection, "normal": normal.copy(),
                  "duplicate": duplicate, "contact": contact})
  return seeds


def _capture_staged_workspace(sim):
  """Wrap stage calls with read-only same-stream copies of their workspace."""
  owner = sim._coupled_constraints
  workspace = owner._workspace
  epochs = []
  current = None
  names = (
      "_contact_sdf_seed_init", "_contact_sdf_descent_prepare",
      "_contact_sdf_line_search", "_contact_sdf_phase_reset",
      "_contact_sdf_publish_normal", "_contact_sdf_publish_contact",
      "_contact_sdf_finalize_producer")
  for name in names:
    original = getattr(owner, name)
    if original is None:
      continue

    def wrapped(*args, _original=original, _name=name, **kwargs):
      nonlocal current
      if _name == "_contact_sdf_seed_init":
        if current is not None:
          raise AssertionError("producer started a new seed tile before finalize")
        current = []
      result = _original(*args, **kwargs)
      dims = owner._sdf_stage_dims.clone()
      # This fixture has one pair and eight seeds. Capture only its fixed
      # eight-seed tile; avoid per-dispatch host synchronization on stage dims.
      state = workspace["contact_sdf_seed_state"][:8 * _STATE_WORDS]
      ctrl = workspace["contact_sdf_seed_ctrl"][:8]
      valid = workspace["contact_sdf_seed_valid"][:8]
      records = workspace["contact_sdf_seed_records"][:8 * 25]
      if current is None:
        raise AssertionError(f"SDF stage outside producer epoch: {_name}")
      current.append((_name, dims, state.clone(), ctrl.clone(), valid.clone(),
                      records.clone()))
      if _name == "_contact_sdf_finalize_producer":
        epochs.append(tuple(current))
        current = None
      return result

    setattr(owner, name, wrapped)
  return epochs, lambda: current


def _host_capture(snapshot):
  return tuple(value.detach().cpu().numpy().copy()
               if hasattr(value, "detach") else value for value in snapshot)


def _collapse_words(words):
  # SDF wide values are serialized as three coordinate planes: hi.xyz,
  # lo.xyz, tail.xyz. Sum planes, not coordinates within each plane.
  return np.asarray(words, dtype=np.float64).reshape(3, 3).sum(axis=0)


def _collapse_wide(words):
  return float(np.asarray(words, dtype=np.float64).sum())


def _mismatch_detail(got, expected):
  got = np.asarray(got, dtype=np.float64)
  expected = np.asarray(expected, dtype=np.float64)
  difference = np.abs(got - expected)
  spacing = np.abs(np.spacing(expected))
  ulps = difference / np.maximum(spacing, np.finfo(np.float64).tiny)
  return {"got_hex": [float(x).hex() for x in got.reshape(-1)],
          "source_hex": [float(x).hex() for x in expected.reshape(-1)],
          "max_abs": float(np.max(difference, initial=0.0)),
          "max_source_ulps": float(np.max(ulps, initial=0.0))}


def _select_complete_epoch(epochs):
  complete = [epoch for epoch in epochs if epoch
              and epoch[0][0] == "_contact_sdf_seed_init"
              and epoch[-1][0] == "_contact_sdf_finalize_producer"]
  if not complete:
    raise AssertionError("no complete SDF producer epoch")
  return complete[-1]


def _event_rank(seed, stage, positions):
  """Order divergence by dispatch before lane, rather than seed-loop order."""
  prefix, _, field = stage.partition(".")
  if prefix == "seed_init":
    key = ("_contact_sdf_seed_init", 2, 0)
  elif prefix == "publish_contact":
    key = ("_contact_sdf_publish_contact", 0, 0)
  else:
    pass_name, iteration = prefix.rstrip("]").split("[")
    mode = 2 if pass_name == "collision" else 0
    prepare = field in ("x0", "gradient", "dist0", "alpha0", "wolfe0", "capture")
    key = ("_contact_sdf_descent_prepare" if prepare
           else "_contact_sdf_line_search", mode, int(iteration))
  return positions.get(key, len(positions)), seed


def _compare_trace(name, reference, captures, native_contacts, input_digest):
  """Assert source/native seed state agreement and report first divergence."""
  assert captures, (name, "no complete SDF producer epoch")
  host_epochs = [[_host_capture(snapshot) for snapshot in epoch]
                 for epoch in captures]
  # Select one whole epoch. Never mix the initialization, iteration, and
  # publication states from different assembled_system invocations.
  host = _select_complete_epoch(host_epochs)
  assert host[0][0] == "_contact_sdf_seed_init"
  assert host[-1][0] == "_contact_sdf_finalize_producer"
  assert sum(stage == "_contact_sdf_seed_init" for stage, *_ in host) == 1
  assert sum(stage == "_contact_sdf_finalize_producer"
             for stage, *_ in host) == 1
  assert int(host[0][1][0]) == 8 and int(host[0][1][1]) == 0
  assert all(int(dims[0]) == 8 and int(dims[1]) == 0
             for _, dims, *_ in host)
  # One epoch has one dispatch per source stage. Duplicate keys would mean the
  # producer crossed a tile/epoch boundary and the fixture bounds were wrong.
  by_key = {}
  positions = {}
  for stage, dims, state, ctrl, valid, records in host:
    tile_count, seed_start, capacity, mode, step = map(int, dims.tolist())
    key = (stage, mode, step, seed_start, tile_count)
    assert key not in by_key, (name, "duplicate stage in producer epoch", key)
    positions[(stage, mode, step)] = len(positions)
    by_key[key] = (state, ctrl, valid, records)
  first_exact = None
  first_tolerance = None
  first_branch = None

  def compare(seed, stage, got, expected, atol):
    nonlocal first_exact, first_tolerance
    detail = _mismatch_detail(got, expected)
    rank = _event_rank(seed, stage, positions)
    if detail["max_abs"] != 0.0 and (first_exact is None or
        rank < _event_rank(first_exact[0], first_exact[1], positions)):
      first_exact = (seed, stage, detail)
    if detail["max_abs"] > atol and (first_tolerance is None or
        rank < _event_rank(first_tolerance[0], first_tolerance[1], positions)):
      first_tolerance = (seed, stage, detail, atol)

  for seed in reference:
    index = int(seed["seed"])
    init = by_key.get(("_contact_sdf_seed_init", 2, 0, 0,
                       min(32, len(reference))))
    assert init is not None, (name, "missing initialization capture")
    state = init[0].reshape(-1, _STATE_WORDS)[index]
    assert int(init[2][index]) == 1, (name, index, "seed validity")
    got_x = _collapse_words(state[_STAGE["x"]:_STAGE["x"] + 9])
    compare(index, "seed_init.x", got_x, seed["x_init"], 2e-6)
    for mode, pass_name in ((2, "collision"), (0, "intersection")):
      history = seed[pass_name]
      for iteration, expected in enumerate(history):
        prep = by_key.get(("_contact_sdf_descent_prepare", mode, iteration,
                           0, min(32, len(reference))))
        line = by_key.get(("_contact_sdf_line_search", mode, iteration,
                           0, min(32, len(reference))))
        if prep is None or line is None:
          first_branch = (index, f"{pass_name}[{iteration}].capture", prep, line)
          break
        pstate = prep[0].reshape(-1, _STATE_WORDS)[index]
        got_x0 = _collapse_words(pstate[_STAGE["x0"]:_STAGE["x0"] + 9])
        compare(index, f"{pass_name}[{iteration}].x0", got_x0,
                expected["x0"], 3e-5)
        got_grad = _collapse_words(pstate[_STAGE["grad"]:
                                          _STAGE["grad"] + 9])
        expected_grad = expected["grad"]
        compare(index, f"{pass_name}[{iteration}].gradient", got_grad,
                expected_grad, 3e-5)
        got_dist0 = _collapse_wide(pstate[_STAGE["dist0"]:
                                          _STAGE["dist0"] + 3])
        compare(index, f"{pass_name}[{iteration}].dist0", [got_dist0],
                [expected["dist0"]], 3e-6)
        for field in ("alpha0", "wolfe0"):
          offset = _STAGE[field[:-1]]
          got = _collapse_wide(pstate[offset:offset + 3])
          compare(index, f"{pass_name}[{iteration}].{field}", [got],
                  [expected[field]], 3e-6)
        lstate, lctrl, _, _ = line
        lstate = lstate.reshape(-1, _STATE_WORDS)[index]
        got_x = _collapse_words(lstate[_STAGE["x"]:
                                       _STAGE["x"] + 9])
        for field in ("alpha", "wolfe", "dist"):
          got = _collapse_wide(lstate[_STAGE[field]:_STAGE[field] + 3])
          compare(index, f"{pass_name}[{iteration}].{field}", [got],
                  [expected[field]], 3e-6)
        expected_ctrl = int(not (expected["dist0"] < expected["dist"]))
        got_ctrl = int(lctrl[index])
        branch_stage = f"{pass_name}[{iteration}].ctrl"
        if got_ctrl != expected_ctrl and (first_branch is None or
            _event_rank(index, branch_stage, positions) <
            _event_rank(first_branch[0], first_branch[1], positions)):
          first_branch = (index, f"{pass_name}[{iteration}].ctrl",
                          got_ctrl, expected_ctrl)
        compare(index, f"{pass_name}[{iteration}].x", got_x,
                expected["x"], 3e-5)
  if first_tolerance or first_branch:
    pytest.fail(f"{name}: first tolerance divergence={first_tolerance!r}; "
                f"first branch divergence={first_branch!r}; "
                f"earliest exact bit/ULP difference={first_exact!r}; "
                f"input_sha256={input_digest}")

  publish = by_key.get(("_contact_sdf_publish_contact", 0, 0, 0,
                        min(32, len(reference))))
  assert publish is not None, (name, "missing contact publication capture")
  seed_records = publish[3].reshape(-1, 25)
  for seed in reference:
    index = int(seed["seed"])
    expected = seed["contact"]
    if expected is None:
      continue
    record = seed_records[index]
    compare(index, "publish_contact.dist", [record[12]],
            [expected["dist"]], 3e-6)
    compare(index, "publish_contact.pos", record[13:16], expected["pos"],
            3e-5)
    compare(index, "publish_contact.normal", record[16:19],
            expected["normal"], 3e-5)

  if first_tolerance:
    pytest.fail(f"{name}: contact publication divergence={first_tolerance!r}; "
                f"earliest exact bit/ULP difference={first_exact!r}; "
                f"input_sha256={input_digest}")

  expected_contacts = [seed["contact"] for seed in reference
                       if seed["contact"] is not None]
  assert len(expected_contacts) == len(native_contacts), (
      name, len(expected_contacts), len(native_contacts))


def _snapshot_available_tensors(container, names):
  return {name: getattr(container, name).clone()
          for name in names
          if hasattr(container, name)
          and hasattr(getattr(container, name), "clone")}


def test_cpu_source_replay_tracks_pinned_bundled_sdf_seed_order():
  """Keep the CPU replay anchored to the installed pinned 3.10 callbacks."""
  for name in _PLUGIN_NAMES:
    model = _plugin_sphere_model(name)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(model.qpos0, dtype=np.float32).astype(np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    sdf_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
        int(mujoco.mjtGeom.mjGEOM_SDF))[0])
    analytic_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
        int(mujoco.mjtGeom.mjGEOM_SPHERE))[0])
    trace = _source_pair_replay(model, data, sdf_geom, analytic_geom)
    source_contacts = [(float(data.contact[i].dist),
                        np.asarray(data.contact[i].frame[:3]),
                        np.asarray(data.contact[i].pos))
                       for i in range(data.ncon)]
    expected = [seed["contact"] for seed in trace
                if seed["contact"] is not None]
    assert len(expected) == len(source_contacts) == 8
    for contact, source in zip(expected, source_contacts):
      np.testing.assert_allclose(contact["dist"], source[0], rtol=0,
                                 atol=3e-6)
      np.testing.assert_allclose(contact["normal"], source[1], rtol=0,
                                 atol=3e-6)
      np.testing.assert_allclose(contact["pos"], source[2], rtol=0,
                                 atol=3e-6)


def test_trace_decoder_sums_wide_planes_and_keeps_latest_trial():
  # Distinct coordinates/planes prevent a mistaken row-wise reduction from
  # passing by symmetry.
  words = np.arange(1.0, 10.0, dtype=np.float64)
  np.testing.assert_array_equal(_collapse_words(words), [12., 15., 18.])
  epochs = [
      [("_contact_sdf_seed_init", "old"),
       ("_contact_sdf_finalize_producer", "old")],
      [("_contact_sdf_seed_init", "new"),
       ("_contact_sdf_finalize_producer", "new")],
  ]
  assert _select_complete_epoch(epochs) == epochs[1]


def test_cpu_pinned_stepgradient_rejection_returns_final_trial_and_distance():
  """A real bowl callback reaches amin and agrees with pinned CPU contacts."""
  model = _plugin_sphere_model("bowl")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  sdf_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
      int(mujoco.mjtGeom.mjGEOM_SDF))[0])
  analytic_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
      int(mujoco.mjtGeom.mjGEOM_SPHERE))[0])
  trace = _source_pair_replay(model, data, sdf_geom, analytic_geom)

  witness = next(entry for entry in trace[0]["collision"]
                 if entry["dist0"] < entry["dist"])
  assert witness["alpha"] == 2.0 ** -14
  assert not np.array_equal(witness["x"], witness["x0"])
  expected_trial = np.zeros(3, dtype=np.float64)
  mujoco.mju_addScl3(expected_trial, witness["x0"], witness["grad"],
                     -witness["alpha"])
  np.testing.assert_array_equal(witness["x"], expected_trial)
  assert trace[0]["collision_dist"] == witness["dist"]
  assert data.ncon == 8 and trace[0]["contact"] is not None
  np.testing.assert_allclose(trace[0]["contact"]["dist"],
                             data.contact[0].dist, rtol=0, atol=3e-6)
  np.testing.assert_allclose(trace[0]["contact"]["pos"],
                             data.contact[0].pos, rtol=0, atol=3e-6)
  np.testing.assert_allclose(trace[0]["contact"]["normal"],
                             data.contact[0].frame[:3], rtol=0, atol=3e-6)


def test_both_rigid_sdf_descenders_preserve_rejected_trial_source_contract():
  source = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/sdf_narrowphase.metal").read_text()
  inline = _source_function(source, "inline SdfWide sdf_descend(")
  staged = _source_function(source, "inline void sdf_stage_line_search(")
  assert "if (sdf_jet_gt(accepted_dist, dist0)) {\n      // Pinned stepGradient" in inline
  assert "return sjwide_value(accepted_dist);" in inline
  assert "x = x0;" not in inline
  assert "bool increased = sdf_jet_gt(accepted_dist, dist0);" in staged
  assert "SdfWide dist = sjwide_value(accepted_dist);" in staged
  assert "if (increased) x = x0;" not in staged
  positions = {("_contact_sdf_seed_init", 2, 0): 0,
               ("_contact_sdf_line_search", 2, 5): 12}
  assert _event_rank(7, "seed_init.x", positions) < _event_rank(
      0, "collision[5].x", positions)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native per-seed SDF source trace")
@pytest.mark.parametrize("name", _PLUGIN_NAMES)
def test_native_staged_sdf_matches_pinned_per_seed_source_trace(name):
  model = _plugin_sphere_model(name)
  assert model.npair == 1 and int(model.opt.sdf_initpoints) == 8
  input_qpos = np.asarray(model.qpos0, dtype=np.float32)[None, :]
  input_qvel = np.zeros((1, model.nv), dtype=np.float32)
  control = MetalSimulation(model, batch_size=1, qpos=input_qpos,
                            qvel=input_qvel,
                            profile="integrated_euler_v1")
  sim = MetalSimulation(model, batch_size=1, qpos=input_qpos,
                        qvel=input_qvel, profile="integrated_euler_v1")
  represented_qpos = sim.state._qpos[0].detach().cpu().numpy().astype(
      np.float64, copy=True)
  represented_qvel = sim.state._qvel[0].detach().cpu().numpy().astype(
      np.float64, copy=True)
  np.testing.assert_array_equal(represented_qpos, input_qpos[0].astype(np.float64))
  np.testing.assert_array_equal(represented_qvel, input_qvel[0].astype(np.float64))
  input_digest = hashlib.sha256(
      np.concatenate((represented_qpos, represented_qvel)).tobytes()).hexdigest()
  source = mujoco.MjData(model)
  source.qpos[:] = represented_qpos
  source.qvel[:] = represented_qvel
  mujoco.mj_forward(model, source)
  sdf_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
      int(mujoco.mjtGeom.mjGEOM_SDF))[0])
  analytic_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
      int(mujoco.mjtGeom.mjGEOM_SPHERE))[0])
  reference = _source_pair_replay(model, source, sdf_geom, analytic_geom)

  state_names = ("_qpos", "_qvel", "_act", "_ctrl", "_warmstart",
                 "_qacc", "_qfrc_constraint", "_status", "_qacc_warmstart")
  control_system = control.assembled_system(recompute=True)
  control_state = _snapshot_available_tensors(control.state, state_names)
  control_warm = _snapshot_available_tensors(
      control, ("_warmstart", "_qacc_warmstart"))
  control_solver_warm = control._coupled_constraints.get_warmstart()
  captures, pending_epoch = _capture_staged_workspace(sim)
  system = sim.assembled_system(recompute=True)
  assert pending_epoch() is None, (name, "incomplete producer epoch")
  state_after = _snapshot_available_tensors(sim.state, state_names)
  sim_after = _snapshot_available_tensors(
      sim, ("_warmstart", "_qacc_warmstart"))
  assert control_state.keys() == state_after.keys()
  assert control_warm.keys() == sim_after.keys()
  for field in control_state:
    assert control_state[field].equal(state_after[field]), (
        name, "observer changed simulation state", field)
  for field in control_warm:
    assert control_warm[field].equal(sim_after[field]), (
        name, "observer changed warmstart state", field)
  np.testing.assert_array_equal(
      sim._coupled_constraints.get_warmstart(), control_solver_warm)
  inert_fields = ("status", "contact_mask", "contact_distance",
                  "contact_normal", "contact_position", "J", "R", "ar",
                  "rhs", "W", "qacc", "qfrc_constraint")
  for field in inert_fields:
    assert field in system and field in control_system, field
    actual = system[field].detach().cpu().numpy()
    plain = control_system[field].detach().cpu().numpy()
    assert np.array_equal(actual, plain), (
        name, "observer changed published system", field,
        hashlib.sha256(actual.tobytes()).hexdigest(),
        hashlib.sha256(plain.tobytes()).hexdigest())
  for field in ("warmstart", "lambda", "qacc_warmstart"):
    if field in system or field in control_system:
      assert field in system and field in control_system
      assert np.array_equal(system[field].detach().cpu().numpy(),
                            control_system[field].detach().cpu().numpy()), (
          name, "observer changed solver warmstart output", field)
  mask = system["contact_mask"][0].detach().cpu().numpy() > 0.5
  native_contacts = [(float(d), n, p) for d, n, p in zip(
      system["contact_distance"][0].detach().cpu().numpy()[mask],
      system["contact_normal"][0].detach().cpu().numpy()[mask],
      system["contact_position"][0].detach().cpu().numpy()[mask])]
  _compare_trace(name, reference, captures, native_contacts, input_digest)
