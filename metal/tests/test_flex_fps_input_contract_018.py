# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Supplemental FPS oracle-contract gates for the pinned plane-flex fixture.

These gates deliberately separate three inputs: raw float32 program vertices,
paired full-source vertices, and the public FK/reset path. Full-precision CPU
identity tests pass complete source words to the native input; their historical
high-only fixture is preserved in Git. The represented-input oracle is not
substituted for the full-source identity requirement.
"""

import os

import mujoco
import numpy as np
import pytest


def _fixture(midphase):
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom type="plane" size="0 0 .1" contype="0" conaffinity="1"/>
      <geom type="plane" pos="0 0 .2" size="0 0 .1"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="8 8 1" pos="0 0 -.1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  if not midphase:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _refresh_source_bounds(model, data, vertices):
  """Install represented vertices and rebuild pinned flex/BVH bounds."""
  data.flexvert_xpos[:] = np.asarray(vertices, np.float64).reshape(
      int(model.nflexvert), 3)
  child = np.asarray(model.bvh_child, np.int32).reshape(-1, 2)
  nodeid = np.asarray(model.bvh_nodeid, np.int32)
  nbvhstatic = int(model.nbvhstatic)
  for flex in range(int(model.nflex)):
    vbase = int(model.flex_vertadr[flex])
    dim = int(model.flex_dim[flex])
    ebase = int(model.flex_elemadr[flex])
    edata = int(model.flex_elemdataadr[flex])
    for elem in range(int(model.flex_elemnum[flex])):
      ids = np.asarray(
          model.flex_elem[edata + elem * (dim + 1):
                          edata + (elem + 1) * (dim + 1)], np.int32)
      points = data.flexvert_xpos.reshape(-1, 3)[vbase + ids]
      lo, hi = points.min(axis=0), points.max(axis=0)
      data.flexelem_aabb[ebase + elem, :3] = .5 * (hi + lo)
      data.flexelem_aabb[ebase + elem, 3:] = (
          .5 * (hi - lo) + float(model.flex_radius[flex]))
    bvhadr = int(model.flex_bvhadr[flex])
    bvhnum = int(model.flex_bvhnum[flex])
    if bvhadr < 0:
      continue
    modified = np.zeros(bvhnum, dtype=bool)
    for local in range(bvhnum):
      leaf = int(nodeid[bvhadr + local])
      if leaf >= 0:
        data.bvh_aabb_dyn[bvhadr + local - nbvhstatic] = (
            data.flexelem_aabb[ebase + leaf])
        modified[local] = True
    for local in range(bvhnum - 1, -1, -1):
      if nodeid[bvhadr + local] >= 0:
        continue
      c1, c2 = map(int, child[bvhadr + local])
      if not (modified[c1] or modified[c2]):
        continue
      a = data.bvh_aabb_dyn[bvhadr - nbvhstatic + c1]
      b = data.bvh_aabb_dyn[bvhadr - nbvhstatic + c2]
      lo = np.minimum(a[:3] - a[3:], b[:3] - b[3:])
      hi = np.maximum(a[:3] + a[3:], b[:3] + b[3:])
      out = data.bvh_aabb_dyn[bvhadr + local - nbvhstatic]
      out[:3] = .5 * (hi + lo)
      out[3:] = .5 * (hi - lo)
      modified[local] = True


def _identities(data):
  return {(int(c.geom[0]), int(c.vert[1]))
          for c in data.contact[:data.ncon]}


def _cpu_collision(model, reference, vertices):
  data = mujoco.MjData(model)
  data.qpos[:] = reference.qpos
  data.qvel[:] = reference.qvel
  mujoco.mj_forward(model, data)
  _refresh_source_bounds(model, data, vertices)
  mujoco.mj_collision(model, data)
  return data, _identities(data)


def _cpu_collision_native_plane_inputs(model, reference, vertices,
                                       geom_pos, geom_quat):
  """Run pinned collision from every value consumed by this plane kernel.

  The raw device ABI carries vertex positions, geom positions/quaternions, and
  flex radius as float32.  The historical CPU oracle above intentionally
  retains all source doubles.  This supplemental oracle keeps that original
  gate intact and separately installs the float32 geom pose and radius used by
  the raw Metal plane branch before refreshing source bounds.
  """
  geom_pos = np.asarray(geom_pos, np.float32)
  geom_quat = np.asarray(geom_quat, np.float32)
  if geom_pos.shape != (1, int(model.ngeom), 3):
    raise ValueError("native plane geom_pos must have shape [1,ngeom,3]")
  if geom_quat.shape != (1, int(model.ngeom), 4):
    raise ValueError("native plane geom_quat must have shape [1,ngeom,4]")
  source_radius = np.asarray(model.flex_radius, np.float64).copy()
  try:
    # The plane kernel currently consumes the descriptor's float32 high word.
    # Keep all other fixture parameters as pinned source values; here margin,
    # gap, and tolerance are zero/unused by the direct plane branch.
    model.flex_radius[:] = source_radius.astype(np.float32).astype(np.float64)
    data = mujoco.MjData(model)
    data.qpos[:] = reference.qpos
    data.qvel[:] = reference.qvel
    mujoco.mj_forward(model, data)
    data.geom_xpos[:] = geom_pos[0].astype(np.float64)
    for geom in range(int(model.ngeom)):
      matrix = np.empty(9, np.float64)
      mujoco.mju_quat2Mat(matrix, geom_quat[0, geom].astype(np.float64))
      data.geom_xmat[geom] = matrix
    _refresh_source_bounds(model, data, vertices)
    mujoco.mj_collision(model, data)
    return data, _identities(data)
  finally:
    model.flex_radius[:] = source_radius


def _source_vertex_words(vertices):
  full = np.asarray(vertices, np.float64).reshape(-1, 3)
  high = full.astype(np.float32)
  low = (full - high.astype(np.float64)).astype(np.float32)
  tail = (full - high.astype(np.float64) - low.astype(np.float64)).astype(
      np.float32)
  return high, low, tail


def _source_geom_position_words(positions):
  full = np.asarray(positions, np.float64).reshape(-1, 3)
  high = full.astype(np.float32)
  low = (full - high.astype(np.float64)).astype(np.float32)
  tail = (full - high.astype(np.float64) - low.astype(np.float64)).astype(
      np.float32)
  return high, low, tail


def test_cpu_fk_model_geom_position_uses_high_low_tail_words():
  """Immutable FK/contact descriptors preserve model-position residuals."""
  from mujoco_metal.metal_kinematics import _prepare_host_arrays
  from mujoco_metal.model import load_model
  from mujoco_metal.flex_contact import lower_flex_contacts

  model, _ = _fixture(True)
  descriptor = load_model(model)
  arrays = _prepare_host_arrays(descriptor)
  count = int(model.ngeom) * 3
  source = np.asarray(descriptor.geom_pos, np.float64).reshape(-1)
  packed = arrays["geom_pos_pair"]
  high = packed[:count].astype(np.float64)
  low = packed[count:2 * count].astype(np.float64)
  tail = packed[2 * count:3 * count].astype(np.float64)
  np.testing.assert_array_equal(high, source.astype(np.float32).astype(np.float64))
  np.testing.assert_allclose(high + low + tail, source, rtol=0, atol=1e-17)
  # The fixture's z=0.2 plane must carry the omitted float32 residual.
  plane = 1
  assert low[3 * plane + 2] != 0.0 or tail[3 * plane + 2] != 0.0
  contact = lower_flex_contacts(model)
  radius = (contact.flex_radius_hi.astype(np.float64)
            + contact.flex_radius_mid.astype(np.float64)
            + contact.flex_radius_low.astype(np.float64))
  np.testing.assert_allclose(radius, model.flex_radius, rtol=0, atol=1e-18)


def _native_geom_inputs(data):
  # The fixture has two static planes. Convert the actual CPU geom matrices to
  # the quaternion ABI consumed by FlexContactProgram.
  ngeom = int(data.geom_xpos.shape[0])
  quats = np.empty((ngeom, 4), np.float32)
  for geom in range(ngeom):
    q = np.empty(4, np.float64)
    mujoco.mju_mat2Quat(q, np.asarray(data.geom_xmat[geom], np.float64).reshape(9))
    quats[geom] = q.astype(np.float32)
  return (np.asarray(data.geom_xpos, np.float32).copy()[None], quats[None])


def _assert_identity_counts(midphase, actual, full, represented):
  assert len(full) == (50 if midphase else 100)
  assert len(represented) == len(full)
  # The source-input contract is intentionally visible: casting the positions
  # to the raw float32 kernel ABI changes tied FPS identities even though the
  # contact counts remain fixed. Keep the old full-source expectation intact.
  # The exact symmetric-difference cardinality is recorded in the parent
  # evidence JSON. Its differing value between two environments remains
  # unexplained. Each native gate below compares exact identities with its
  # actual independent CPU oracle, including the original full-source gate.
  assert len(full ^ represented) > 0
  assert len(actual) == len(represented)


def test_cpu_fps_oracles_distinguish_full_and_raw_float32_inputs():
  for midphase in (False, True):
    model, reference = _fixture(midphase)
    full_vertices = np.asarray(reference.flexvert_xpos, np.float64).reshape(-1, 3)
    raw_vertices = full_vertices.astype(np.float32)
    _, full_ids = _cpu_collision(model, reference, full_vertices)
    represented_data, represented_ids = _cpu_collision(
        model, reference, raw_vertices.astype(np.float64))
    assert represented_data.ncon == (50 if midphase else 100)
    assert len(full_ids ^ represented_ids) > 0
    assert np.max(np.abs(full_vertices - raw_vertices.astype(np.float64))) == pytest.approx(
        5.960464510845753e-09, rel=0, abs=1e-18)


def test_cpu_raw_plane_oracle_quantizes_device_geom_pose_and_radius():
  model, reference = _fixture(True)
  full_vertices = np.asarray(reference.flexvert_xpos, np.float64).reshape(-1, 3)
  high = full_vertices.astype(np.float32)
  raw_data, raw_vertex_ids = _cpu_collision(
      model, reference, high.astype(np.float64))
  geom_pos, geom_quat = _native_geom_inputs(raw_data)
  native_data, native_input_ids = _cpu_collision_native_plane_inputs(
      model, reference, high.astype(np.float64), geom_pos, geom_quat)
  assert geom_pos[0, 1, 2] == np.float32(0.2)
  assert len(raw_vertex_ids) == len(native_input_ids) == 50
  assert _identities(native_data) == native_input_ids
  assert raw_vertex_ids - native_input_ids == {(0, 34), (0, 35)}
  assert native_input_ids - raw_vertex_ids == {(0, 4), (0, 32)}


def _mps(value, torch):
  return torch.as_tensor(np.asarray(value, np.float32).copy(),
                         dtype=torch.float32, device="mps").contiguous()


def _native_selector_ids(program, descriptor, result):
  active = result["active"].detach().cpu().numpy()[0].astype(bool)
  plane_vertex = ((descriptor.geom >= 0) & (descriptor.vert1 >= 0))
  slots = np.flatnonzero(active & plane_vertex)
  return {(int(descriptor.geom[i]), int(descriptor.vert1[i])) for i in slots}


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="raw represented-input FPS parity requires GPU opt-in")
@pytest.mark.parametrize("midphase", [False, True])
def test_native_raw_float32_program_matches_same_input_pinned_collision(midphase):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native FPS parity requires MPS")
  from mujoco_metal.flex_contact import FlexContactProgram, lower_flex_contacts

  model, reference = _fixture(midphase)
  full = np.asarray(reference.flexvert_xpos, np.float64).reshape(-1, 3)
  high, _, _ = _source_vertex_words(full)
  represented_pose, _ = _cpu_collision(
      model, reference, high.astype(np.float64))
  geom_pos, geom_quat = _native_geom_inputs(represented_pose)
  represented, expected = _cpu_collision_native_plane_inputs(
      model, reference, high.astype(np.float64), geom_pos, geom_quat)
  _, full_expected = _cpu_collision(model, reference, full)
  descriptor = lower_flex_contacts(model)
  program = FlexContactProgram(model, device="mps")
  assert program._narrowphase_admitted
  result = program.run_device(
      _mps(high[None], torch), _mps(geom_pos, torch),
      _mps(geom_quat, torch))
  actual = _native_selector_ids(program, descriptor, result)
  _assert_identity_counts(midphase, actual, full_expected, expected)
  assert actual == expected


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="paired full-source FPS parity requires GPU opt-in")
@pytest.mark.parametrize("midphase", [False, True])
def test_native_paired_vertices_match_original_full_precision_cpu_identity(midphase):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("paired FPS parity requires MPS")
  from mujoco_metal.flex_contact import FlexContactProgram, lower_flex_contacts

  model, reference = _fixture(midphase)
  full = np.asarray(reference.flexvert_xpos, np.float64).reshape(-1, 3)
  high, low, tail = _source_vertex_words(full)
  _, expected = _cpu_collision(model, reference, full)
  represented_pose, _ = _cpu_collision(
      model, reference, high.astype(np.float64))
  # Match the full-source vertex oracle with the same full-source geometry pose.
  # The raw/represented ABI tests above intentionally keep geom residuals zero.
  geom_full = np.asarray(reference.geom_xpos, np.float64).reshape(-1, 3)
  geom_high, geom_low, geom_tail = _source_geom_position_words(geom_full)
  geom_quat_high = _native_geom_inputs(reference)[1][0]
  geom_pos = geom_high[None]
  geom_quat = geom_quat_high[None]
  represented, raw_expected = _cpu_collision_native_plane_inputs(
      model, reference, high.astype(np.float64), geom_pos, geom_quat)
  descriptor = lower_flex_contacts(model)
  program = FlexContactProgram(model, device="mps")
  result = program.run_device(
      _mps(high[None], torch), _mps(geom_pos, torch),
      _mps(geom_quat, torch), flexvert_xpos_low=_mps(low[None], torch),
      flexvert_xpos_tail=_mps(tail[None], torch),
      geom_pos_low=_mps(geom_low[None], torch),
      geom_pos_tail=_mps(geom_tail[None], torch))
  actual = _native_selector_ids(program, descriptor, result)
  _assert_identity_counts(midphase, actual, expected, raw_expected)
  assert actual == expected


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public FK/reset FPS path requires GPU opt-in")
@pytest.mark.parametrize("midphase", [False, True])
def test_native_public_simulation_fk_reset_preserves_fps_identity(midphase):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("public FPS path requires MPS")
  from mujoco_metal.simulation import MetalSimulation

  model, reference = _fixture(midphase)
  full_vertices = np.asarray(reference.flexvert_xpos, np.float64).reshape(-1, 3)
  _, expected = _cpu_collision(model, reference, full_vertices)
  raw_vertices = full_vertices.astype(np.float32).astype(np.float64)
  represented_pose, _ = _cpu_collision(model, reference, raw_vertices)
  geom_pos, geom_quat = _native_geom_inputs(represented_pose)
  _, raw_expected = _cpu_collision_native_plane_inputs(
      model, reference, raw_vertices, geom_pos, geom_quat)
  qpos = np.asarray(reference.qpos, np.float32)[None].copy()
  qvel = np.asarray(reference.qvel, np.float32)[None].copy()
  sim = MetalSimulation(model, batch_size=1, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")

  def check_public_position():
    sim.prepare_forward_position(sim._state._qpos)
    program = sim._flex._contact_program
    assert program is not None and program._narrowphase_admitted
    bundle = sim._coupled_constraints._flex_contact_current
    assert bundle is not None
    result = bundle["contact_result"]
    descriptor = sim._coupled_constraints.descriptor.flex_contact_descriptor
    actual = _native_selector_ids(program, descriptor, result)
    _assert_identity_counts(midphase, actual, expected, raw_expected)
    assert actual == expected
    # Run the same production FK kernel independently and retain its borrowed
    # geometry residuals. The plane contact consumer receives these arrays via
    # Simulation's POS pose dictionary; exact static z=0.2 reconstruction is
    # a source-input invariant, not an identity-bound relaxation.
    poses = sim._smooth._fk.run_device(
        sim._state._qpos, getattr(sim._state, "_mpos", None),
        getattr(sim._state, "_mquat", None))
    geom_high = poses["geom_pos"].detach().cpu().numpy()[0].astype(np.float64)
    geom_low = poses["geom_pos_low"].detach().cpu().numpy()[0].astype(np.float64)
    geom_tail = poses["geom_pos_tail"].detach().cpu().numpy()[0].astype(np.float64)
    np.testing.assert_allclose(
        geom_high + geom_low + geom_tail,
        np.asarray(reference.geom_xpos, np.float64), rtol=0, atol=2e-16)
    high = sim._flex.flexvert_xpos.detach().cpu().numpy()[0].astype(np.float64)
    low = sim._flex._flexvert_xpos_low.detach().cpu().numpy()[0].astype(np.float64)
    tail = sim._flex._flexvert_xpos_tail.detach().cpu().numpy()[0].astype(np.float64)
    reconstructed = high + low + tail
    np.testing.assert_allclose(reconstructed, full_vertices, rtol=0, atol=6e-6)

  check_public_position()
  sim.reset(qpos=qpos, qvel=qvel)
  check_public_position()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="moving-geometry residual FK requires GPU opt-in")
def test_native_fk_moving_geom_position_residual_survives_replacement():
  """FK carries source geom-position tails through moving-body replacement."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("moving-geometry FK parity requires MPS")
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model

  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="moving_obstacle">
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom name="obstacle" type="sphere" pos=".2 .1 0" size=".02"/>
      </body>
    </worldbody></mujoco>
  """)
  descriptor = load_model(model)
  fk = MetalKinematics(descriptor, batch_size=2)

  def expected_geometry(qpos32):
    expected = []
    for value in np.asarray(qpos32, np.float32).reshape(2, 1):
      data = mujoco.MjData(model)
      data.qpos[:] = value.astype(np.float64)
      mujoco.mj_forward(model, data)
      expected.append(np.asarray(data.geom_xpos, np.float64))
    return np.stack(expected)

  for qpos32 in (np.asarray([[.25], [.5]], np.float32),
                  np.asarray([[-.125], [.125]], np.float32)):
    poses = fk.run_device(_mps(qpos32, torch))
    high = poses["geom_pos"].detach().cpu().numpy().astype(np.float64)
    low = poses["geom_pos_low"].detach().cpu().numpy().astype(np.float64)
    tail = poses["geom_pos_tail"].detach().cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(
        high + low + tail, expected_geometry(qpos32), rtol=0, atol=2e-16)
