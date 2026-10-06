# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned convex-mesh versus heightfield collision coverage for REQ-GEO-004."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.capacity import CapacityLimits


_LIMITS = CapacityLimits(max_slots=50, max_rows=600)


_GRID = 9
_ELEVATION = " ".join(["0.5"] * (_GRID * _GRID))
_BOX_VERTICES = " ".join(
    f"{x} {y} {z}" for z in (-0.05, 0.05)
    for y in (-0.12, 0.12) for x in (-0.12, 0.12))
_XML = f'''<mujoco>
  <asset>
    <hfield name="terrain_asset" nrow="{_GRID}" ncol="{_GRID}"
        size=".4 .4 .05 .02" elevation="{_ELEVATION}"/>
    <mesh name="convex_box" vertex="{_BOX_VERTICES}"/>
  </asset>
  <option timestep=".001" integrator="Euler" iterations="100"
      tolerance="1e-8" gravity="0 0 -9.81" cone="elliptic"/>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain_asset"
        contype="1" conaffinity="1" condim="1" friction=".7"/>
    <body name="mesh_body" pos="0 0 .03">
      <freejoint/>
      <geom name="mesh_geom" type="mesh" mesh="convex_box"
          contype="1" conaffinity="1" condim="1" friction=".7"/>
    </body>
  </worldbody>
</mujoco>'''


def _model(*, condim=1, cone="elliptic", xml=None):
  xml = _XML if xml is None else xml
  xml = xml.replace('condim="1"', f'condim="{condim}"')
  xml = xml.replace('cone="elliptic"', f'cone="{cone}"')
  return mujoco.MjModel.from_xml_string(xml)


def _source_contacts(model, *, qpos=None):
  """Pinned oracle at the explicit public pose, or the compiled default."""
  data = mujoco.MjData(model)
  if qpos is not None:
    data.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(model.nq)
  mujoco.mj_forward(model, data)
  terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  return data, [data.contact[i] for i in range(data.ncon)
                if set(map(int, data.contact[i].geom)) == {terrain, mesh}]


def test_mesh_heightfield_source_fixture_has_multiple_pinned_prism_witnesses():
  model = _model()
  desc = lower_coupled_constraints(model, limits=_LIMITS)
  assert desc.npairs == 1
  assert tuple(desc.pair_max_contacts) == (50,)
  data, contacts = _source_contacts(model)
  assert data.ncon >= 2 and len(contacts) >= 2
  assert len(contacts) <= 50
  assert all(float(c.dist) < 0.0 for c in contacts)
  assert all(np.isfinite(np.asarray(c.frame)).all()
             and np.isfinite(np.asarray(c.pos)).all() for c in contacts)
  # This fixture actually spans several terrain prisms. A single hull/GJK
  # contact, a sphere proxy, or an empty native slot count cannot pass.
  assert len({tuple(np.round(np.asarray(c.pos)[:2], 3)) for c in contacts}) >= 4


def test_hfield_prism_support_matches_pinned_strict_source_order_cpu():
  from mujoco_metal.common_ccd_bridge import (
      hfield_prism_support_record, hfield_source_support_index)

  model = _model()
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  _, words, high = hfield_prism_support_record(
      model, geom, 2, 3, 0, 0.0, vertex_offset=0)
  residual = words[48:84].reshape(6, 3, 2)
  vertices = high.reshape(6, 3).astype(np.float64)
  vertices += residual[:, :, 0].astype(np.float64)
  vertices += residual[:, :, 1].astype(np.float64)
  assert hfield_source_support_index(vertices, (1.0, 0.0, -1.0)) in (0, 1, 2)
  assert hfield_source_support_index(vertices, (1.0, 0.0, 0.0)) in (3, 4, 5)
  tied = np.asarray([[0, 0, -1], [1, 0, -1], [2, 0, -1],
                     [0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
  assert hfield_source_support_index(tied, (0, 0, 1)) == 3
  assert hfield_source_support_index(tied, (0, 0, -1)) == 0


def test_hfield_prism_grid_spacing_uses_pinned_mjt_num_order_cpu():
  """Catch the float-step discrepancy in production prism construction."""
  from mujoco_metal.common_ccd_bridge import hfield_prism_support_record

  model = _model()
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  hid = int(model.geom_dataid[geom])
  rows, cols = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
  size = np.asarray(model.hfield_size[hid], dtype=np.float64)
  dx = (2.0 * size[0]) / (cols - 1)
  dy = (2.0 * size[1]) / (rows - 1)
  # Source-order addPrismVert uses double spacing and one C expression
  # `dx*c - size0`; on arm64 clang contracts it to fnmsub. The former
  # float-only path differed by several nanometres for this grid.
  from decimal import Decimal, localcontext
  def source_fma(a, b, c):
    with localcontext() as context:
      context.prec = 120
      return float(Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                   + Decimal.from_float(float(c)))
  source_xy = np.asarray([source_fma(dx, 3, -size[0]),
                          source_fma(dy, 3, -size[1])])
  float_dx = np.float32(2.0 * np.float32(size[0]) / np.float32(cols - 1))
  float_dy = np.float32(2.0 * np.float32(size[1]) / np.float32(rows - 1))
  old_float_xy = np.asarray([
      np.float32(float_dx * np.float32(3) - np.float32(size[0])),
      np.float32(float_dy * np.float32(3) - np.float32(size[1])),
  ], dtype=np.float64)
  assert np.max(np.abs(source_xy - old_float_xy)) > 1e-9

  _, words, high = hfield_prism_support_record(
      model, geom, 2, 3, 0, 0.0, vertex_offset=0)
  residual = words[48:84].reshape(6, 3, 2)
  vertices = high.reshape(6, 3).astype(np.float64)
  vertices += residual[:, :, 0].astype(np.float64)
  vertices += residual[:, :, 1].astype(np.float64)
  np.testing.assert_allclose(vertices[5, :2], source_xy, rtol=0, atol=2e-16)
  assert not np.array_equal(vertices[5, :2], old_float_xy)


def test_hfield_compiled_dimensions_retain_source_double_low_words_cpu():
  """Verify production upload keeps each mjtNum size's full 53-bit value."""
  from mujoco_metal.common_ccd_bridge import build_common_ccd_model_upload

  model = _model()
  legacy_vertices = np.zeros(0, dtype=np.float32)
  legacy_info = np.zeros(9 * int(model.ngeom), dtype=np.int32)
  mesh_hull, info, layout = build_common_ccd_model_upload(
      model, legacy_vertices, legacy_info)
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  record = int(layout["geom_base"]) + 9 * geom
  size_base = int(info[record + 5])
  source_size = np.asarray(
      model.hfield_size[int(model.geom_dataid[geom])], dtype=np.float64)
  high = source_size.astype(np.float32)
  remainder = source_size - high.astype(np.float64)
  middle = remainder.astype(np.float32)
  tail = (remainder - middle.astype(np.float64)).astype(np.float32)
  packed = mesh_hull[size_base:size_base + 12].reshape(4, 3)
  np.testing.assert_array_equal(packed[:, 0], high)
  np.testing.assert_array_equal(packed[:, 1], middle)
  np.testing.assert_array_equal(packed[:, 2], tail)
  assert np.any(middle != 0.0) and np.any(tail != 0.0)
  reconstructed = packed[:, 0].astype(np.float64)
  reconstructed += packed[:, 1].astype(np.float64)
  reconstructed += packed[:, 2].astype(np.float64)
  np.testing.assert_array_equal(reconstructed, source_size)


def test_compiled_mesh_support_preserves_cached_first_max_cpu():
  from mujoco_metal.common_ccd_bridge import compiled_mesh_source_support_index

  vertices = np.asarray([[-1, 0, 0], [1, 0, 0], [1, 0, 0],
                         [0, 0, 1]], dtype=np.float32)
  assert compiled_mesh_source_support_index(vertices, (0, 1, 0)) == 0
  assert compiled_mesh_source_support_index(vertices, (1, 0, 0)) == 1
  assert compiled_mesh_source_support_index(vertices, (1, 0, 0), 2) == 2


def test_hfield_rolling_add_prism_vertices_match_bridge_records_cpu():
  """Check candidate triangles against pinned addPrismVert rolling order."""
  from mujoco_metal.common_ccd_bridge import hfield_prism_support_record

  for kind in ("flat", "slope", "ridge"):
    xml = _XML
    if kind == "slope":
      values = [j / (_GRID - 1) for i in range(_GRID) for j in range(_GRID)]
      xml = xml.replace(_ELEVATION, " ".join(map(str, values)))
    elif kind == "ridge":
      values = [1.0 if abs(j - 4) <= 1 else 0.1
                for i in range(_GRID) for j in range(_GRID)]
      xml = xml.replace(_ELEVATION, " ".join(map(str, values)))
    model = mujoco.MjModel.from_xml_string(xml)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    hid = int(model.geom_dataid[geom])
    rows, cols = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    size = np.asarray(model.hfield_size[hid], dtype=np.float64)
    adr = int(model.hfield_adr[hid])
    data = np.asarray(model.hfield_data[adr:adr + rows * cols], dtype=np.float64)
    margin = 0.013
    dx = (2.0 * size[0]) / (cols - 1)
    dy = (2.0 * size[1]) / (rows - 1)

    for row in range(rows - 1):
      prism = np.zeros((6, 3), dtype=np.float64)
      prism[:3, 2] = -size[3]

      from decimal import Decimal, localcontext
      def source_fma(a, b, c):
        with localcontext() as context:
          context.prec = 120
          return float(Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                       + Decimal.from_float(float(c)))

      def add_prism_vertex(col, tri):
        prism[0] = prism[1]
        prism[1] = prism[2]
        prism[3] = prism[4]
        prism[4] = prism[5]
        dr = 1 - tri
        prism[2, 0] = prism[5, 0] = source_fma(dx, col, -size[0])
        prism[2, 1] = prism[5, 1] = source_fma(dy, row + dr, -size[1])
        z = data[(row + dr) * cols + col] * size[2]
        prism[5, 2] = z + margin

      add_prism_vertex(0, 0)
      add_prism_vertex(0, 1)
      for col in range(1, cols):
        for triangle in (0, 1):
          add_prism_vertex(col, triangle)
          packed, words, high = hfield_prism_support_record(
              model, geom, row, col, triangle, margin, vertex_offset=0)
          assert packed[0] == 8 and packed[4] == 6
          residual = words[48:84].reshape(6, 3, 2)
          rebuilt = high.reshape(6, 3).astype(np.float64)
          rebuilt += residual[:, :, 0].astype(np.float64)
          rebuilt += residual[:, :, 1].astype(np.float64)
          np.testing.assert_array_equal(rebuilt, prism)


@pytest.mark.parametrize("condim,cone,expected_rows", [
    (1, "elliptic", 58), (3, "elliptic", 158),
    (3, "pyramidal", 208), (6, "elliptic", 308),
    (6, "pyramidal", 508),
])
def test_mesh_heightfield_full_source_contact_budget(condim, cone, expected_rows):
  desc = lower_coupled_constraints(
      _model(condim=condim, cone=cone), limits=_LIMITS)
  assert desc.ncontacts_max == 50
  assert desc.nr == expected_rows


@pytest.mark.parametrize("kind", ["slope", "ridge", "pit", "scaled",
                                   "border", "rotated"])
def test_mesh_heightfield_source_transform_and_surface_fixtures(kind):
  xml = _XML
  if kind in ("slope", "ridge", "pit"):
    if kind == "slope":
      values = [j / (_GRID - 1) for i in range(_GRID) for j in range(_GRID)]
    elif kind == "ridge":
      values = [1.0 if abs(j - 4) <= 1 else 0.1
                for i in range(_GRID) for j in range(_GRID)]
    else:
      values = [0.0 if (i - 4) ** 2 + (j - 4) ** 2 < 5 else 0.5
                for i in range(_GRID) for j in range(_GRID)]
    xml = xml.replace(_ELEVATION, " ".join(map(str, values)))
    if kind == "slope":
      xml = xml.replace('pos="0 0 .03"', 'pos="0 0 .045"')
  elif kind == "scaled":
    xml = xml.replace('size=".4 .4 .05 .02"',
                      'size=".55 .3 .08 .03"')
  elif kind == "border":
    xml = xml.replace('pos="0 0 .03"', 'pos=".34 0 .03"')
  elif kind == "rotated":
    xml = xml.replace('pos="0 0 .03"',
                      'pos="0 0 .03" euler="0.1 0.2 0.3"')
  model = mujoco.MjModel.from_xml_string(xml)
  desc = lower_coupled_constraints(model, limits=_LIMITS)
  data, contacts = _source_contacts(model)
  assert desc.npairs == 1 and desc.ncontacts_max == 50
  assert len(contacts) >= 2, kind
  assert np.isfinite([c.dist for c in contacts]).all()
  assert all(float(c.dist) < 0.0 for c in contacts), kind
  assert len({tuple(np.round(np.asarray(c.pos)[:2], 3))
              for c in contacts}) >= 4, kind


def _native_contacts(assembly):
  mask = assembly["contact_mask"].detach().cpu().numpy()[0] > 0.5
  distance = assembly["contact_distance"].detach().cpu().numpy()[0]
  position = assembly["contact_position"].detach().cpu().numpy()[0]
  normal = assembly["contact_normal"].detach().cpu().numpy()[0]
  return [(float(distance[i]), position[i].copy(), normal[i].copy())
          for i in range(len(mask)) if mask[i]]


def _raw_common_contacts(program):
  """Decode the producer records before contact_normal/row assembly."""
  layout = program._common_ccd_layout
  assert layout is not None
  raw = program._constants["mesh_hull"].detach().cpu().numpy()
  info = program._constants["mesh_hull_info"].detach().cpu().numpy()
  header = int(layout["header_base"])
  assert int(info[header]) == 1128487732
  output = int(info[header + 5])
  stride = int(info[header + 6])
  count = int(raw[output])
  status = int(raw[output + 1])
  records = []
  for slot in range(count):
    base = output + 2 + 16 * slot
    records.append((float(raw[base]), raw[base + 4:base + 7].copy(),
                    raw[base + 1:base + 4].copy()))
  assert stride >= 2 + 16 * count
  return status, records


def _raw_common_prism_ids(program):
  """Read diagnostic (row, column, triangle) tags from unused record words."""
  layout = program._common_ccd_layout
  assert layout is not None
  raw = program._constants["mesh_hull"].detach().cpu().numpy()
  output = int(program._constants["mesh_hull_info"].detach().cpu().numpy()[
      int(layout["header_base"]) + 5])
  count = int(raw[output])
  return np.asarray([
      raw[output + 2 + 16 * slot + 13:output + 2 + 16 * slot + 16]
      for slot in range(count)], dtype=np.float32).astype(np.int32)


def _assert_raw_common_matches_rows(produced, consumed):
  assert len(produced) == len(consumed)
  for index, (before, after) in enumerate(zip(produced, consumed)):
    assert before[0] == after[0], index
    np.testing.assert_array_equal(before[1], after[1],
                                  err_msg=f"row position {index}")
    np.testing.assert_array_equal(before[2], after[2],
                                  err_msg=f"row normal {index}")


def test_common_prepass_trace_decoder_matches_synthetic_public_rows():
  produced = [(np.float32(-0.25), np.asarray([1, 2, 3], np.float32),
               np.asarray([0, 0, 1], np.float32))]
  consumed = [(float(produced[0][0]), produced[0][1].copy(),
               produced[0][2].copy())]
  _assert_raw_common_matches_rows(produced, consumed)
  consumed[0][1][0] = np.nextafter(np.float32(1), np.float32(2))
  with pytest.raises(AssertionError):
    _assert_raw_common_matches_rows(produced, consumed)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in common HField producer/consumer trace")
def test_mesh_heightfield_raw_common_witnesses_match_source_and_rows_gpu():
  """Pinpoint whether a strict witness error originates before row assembly."""
  model = _model(condim=1, cone="elliptic")
  public_qpos = np.asarray(model.qpos0, dtype=np.float32)
  _, cpu_contacts = _source_contacts(model, qpos=public_qpos)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1",
                        limits=_LIMITS)
  sim.reset(qpos=public_qpos[None],
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  assembly = sim.assembled_system(recompute=True)
  status, produced = _raw_common_contacts(sim._coupled_constraints)
  prism_ids = _raw_common_prism_ids(sim._coupled_constraints)
  consumed = _native_contacts(assembly)
  assert status == 0
  assert len(produced) == len(consumed) == len(cpu_contacts)
  assert len(prism_ids) == len(produced)
  print("COMMON_HFIELD_PRISM_IDS", prism_ids.tolist())
  assert np.all((prism_ids[:, 0] >= 0) & (prism_ids[:, 0] < _GRID - 1))
  assert np.all((prism_ids[:, 1] >= 1) & (prism_ids[:, 1] < _GRID))
  assert np.all((prism_ids[:, 2] >= 0) & (prism_ids[:, 2] < 2))
  assert all(tuple(prism_ids[i]) < tuple(prism_ids[i + 1])
             for i in range(len(prism_ids) - 1))
  # Producer records are already transformed to world coordinates.  Exact
  # equality here establishes that contact-normal and row assembly preserve
  # the prepass witness; source comparisons locate any remaining GJK/EPA gap.
  _assert_raw_common_matches_rows(produced, consumed)
  for index, (before, source) in enumerate(zip(produced, cpu_contacts)):
    assert abs(before[0] - float(source.dist)) <= 6e-6, index
    np.testing.assert_allclose(before[1], source.pos, rtol=0, atol=6e-6,
                               err_msg=f"raw source position {index}")
    np.testing.assert_allclose(before[2], source.frame[:3], rtol=0, atol=6e-5,
                               err_msg=f"raw source normal {index}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native mesh-heightfield qualification")
@pytest.mark.parametrize(("condim", "cone"), [(1, "elliptic"),
                                               (3, "pyramidal")])
def test_mesh_heightfield_native_contacts_rows_and_checkpoint_replay_gpu(
    condim, cone):
  model = _model(condim=condim, cone=cone)
  public_qpos = np.asarray(model.qpos0, dtype=np.float32)
  desc = lower_coupled_constraints(model, limits=_LIMITS)
  assert desc.npairs == 1 and desc.ncontacts_max == 50
  cpu, cpu_contacts = _source_contacts(model, qpos=public_qpos)
  assert len(cpu_contacts) >= 2
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1",
                        limits=_LIMITS)
  qpos0 = public_qpos.reshape(1, -1)
  qvel0 = np.zeros((1, model.nv), dtype=np.float32)
  sim.reset(qpos=qpos0, qvel=qvel0)
  cpu_step = mujoco.MjData(model)
  cpu_step.qpos[:] = qpos0[0]
  cpu_step.qvel[:] = qvel0[0]
  mujoco.mj_step(model, cpu_step)
  initial = sim.snapshot()
  assembly = sim.assembled_system()
  native = _native_contacts(assembly)
  assert len(native) == len(cpu_contacts)
  # Pinned mjc_ConvexHField traverses the same clamped (row, column, triangle)
  # prism order for the convex mesh and emits one MPR/native-CCD witness per
  # overlapping prism. Compare each ordered witness, including geometry side.
  for index, (actual, source) in enumerate(zip(native, cpu_contacts)):
    dist, position, normal = actual
    assert abs(dist - float(source.dist)) < 1.5e-3, index
    np.testing.assert_allclose(normal, source.frame[:3], atol=2e-3,
                               err_msg=f"contact {index} normal")
    np.testing.assert_allclose(position, source.pos, atol=2e-3,
                               err_msg=f"contact {index} position")
  assert np.count_nonzero(assembly["J"].detach().cpu().numpy()) > 0
  active = assembly["contact_mask"].detach().cpu().numpy()
  assert np.count_nonzero(active) >= len(cpu_contacts)

  sim.step(1)
  after = {name: getattr(sim.state, "_" + name).detach().cpu().numpy().copy()
           for name in ("qpos", "qvel", "qacc", "time", "status")}
  np.testing.assert_allclose(after["qpos"][0], cpu_step.qpos,
                             atol=3e-5, rtol=2e-5)
  np.testing.assert_allclose(after["qvel"][0], cpu_step.qvel,
                             atol=5e-4, rtol=2e-4)
  np.testing.assert_allclose(after["qacc"][0], cpu_step.qacc,
                             atol=3e-2, rtol=2e-3)
  stepped = sim.snapshot()
  sim.reset(qpos=np.asarray([[0., 0., 1., 1., 0., 0., 0.]], np.float32),
            qvel=np.zeros((1, model.nv), np.float32))
  sim.restore(initial)
  sim.step(1)
  for name, expected in after.items():
    np.testing.assert_array_equal(
        getattr(sim.state, "_" + name).detach().cpu().numpy(), expected,
        err_msg=f"checkpoint replay {name}")
  sim.restore(stepped)
  for name, expected in after.items():
    np.testing.assert_array_equal(
        getattr(sim.state, "_" + name).detach().cpu().numpy(), expected,
        err_msg=f"stepped checkpoint {name}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in strict native mesh-heightfield parity")
def test_mesh_heightfield_native_witness_rows_and_first_force_strict_gpu():
  model = _model(condim=1, cone="elliptic")
  public_qpos = np.asarray(model.qpos0, dtype=np.float32)
  descriptor = lower_coupled_constraints(model, limits=_LIMITS)
  cpu, cpu_contacts = _source_contacts(model, qpos=public_qpos)
  assert len(cpu_contacts) >= 2
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1",
                        limits=_LIMITS)
  sim.reset(qpos=public_qpos[None],
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  native = sim.assembled_system(recompute=True)
  contacts = _native_contacts(native)
  assert len(contacts) == len(cpu_contacts)

  for index, (actual, source) in enumerate(zip(contacts, cpu_contacts)):
    distance, position, normal = actual
    assert abs(distance - float(source.dist)) <= 6e-6, index
    np.testing.assert_allclose(position, source.pos, rtol=0, atol=6e-6,
                               err_msg=f"strict contact position {index}")
    np.testing.assert_allclose(normal, source.frame[:3], rtol=0, atol=6e-5,
                               err_msg=f"strict contact normal {index}")

  # The contact witness order also defines the canonical scalar rows for this
  # condim=1 fixture. Compare the independently assembled pinned CPU rows and
  # their impedance/reference data, then check the first-forward force result.
  source_J = np.asarray(cpu.efc_J).reshape(int(cpu.nefc), model.nv)
  row_start = int(descriptor.nr_joint)
  row_stop = row_start + len(cpu_contacts)
  native_J = native["J"].detach().cpu().numpy()[0, row_start:row_stop]
  np.testing.assert_allclose(native_J, source_J, rtol=0, atol=6e-6,
                             err_msg="canonical mesh-heightfield J")
  native_R = native["R"].detach().cpu().numpy()[0, row_start:row_stop]
  native_ar = native["ar"].detach().cpu().numpy()[0, row_start:row_stop]
  np.testing.assert_allclose(native_R, np.asarray(cpu.efc_R),
                             rtol=1e-4, atol=6e-6,
                             err_msg="canonical mesh-heightfield R")
  np.testing.assert_allclose(native_ar, np.asarray(cpu.efc_aref),
                             rtol=2e-4, atol=6e-5,
                             err_msg="canonical mesh-heightfield ar")
  np.testing.assert_allclose(
      native["qfrc_constraint"].detach().cpu().numpy()[0],
      np.asarray(cpu.qfrc_constraint), rtol=2e-3, atol=3e-2,
      err_msg="first-forward mesh-heightfield constraint force")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native narrowphase status propagation")
def test_mesh_heightfield_common_ccd_failure_is_not_silently_empty_contacts():
  """Corrupt bridge metadata must fail the world and recover after repair."""
  model = _model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1",
                        limits=_LIMITS)
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32)[None],
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  program = sim._coupled_constraints
  assert program is not None and program._common_ccd_layout is not None
  info = program._constants["mesh_hull_info"]
  header = 9 * int(model.ngeom)
  valid_magic = int(program._common_ccd_mesh_hull_info[header])
  assert valid_magic == 1128487732

  # This takes the real production candidate kernel's malformed-descriptor
  # branch. Its nonzero status must survive contact-row assembly and the
  # solver dispatch, rather than being interpreted as an ordinary no-contact
  # result.
  info[header] = 0
  failed = sim.assembled_system(recompute=True)
  assert failed["status"].cpu().numpy().tolist() == [2]
  assert not failed["contact_mask"].cpu().numpy().any()

  info[header] = valid_magic
  recovered = sim.assembled_system(recompute=True)
  assert recovered["status"].cpu().numpy().tolist() == [0]
  assert len(_native_contacts(recovered)) == len(_source_contacts(model)[1])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in per-world common CCD failure isolation")
def test_mesh_heightfield_bad_world_ccd_status_preserves_healthy_neighbor():
  import torch

  model = _model(condim=1, cone="elliptic")
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1",
                        limits=_LIMITS)
  sim.reset(qpos=np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2,
                           axis=0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  state = sim.state
  dynamics = sim._smooth.run_device(
      state._qpos, state._qvel, state._mpos, state._mquat)
  poses = dict(dynamics["poses"])
  poses["geom_quat"] = poses["geom_quat"].clone()
  mesh_geom = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  # The mesh position still overlaps the terrain, so broadphase retains the
  # pair. The invalid quaternion reaches the actual common candidate kernel
  # only in world zero; world one has the same valid source pose as CPU.
  poses["geom_quat"][0, mesh_geom, 0] = float("nan")
  program = sim._coupled_constraints
  program.generate_candidates(poses, state._qvel)
  status = program._workspace["out_status"].detach().cpu().numpy().copy()
  assert status.tolist() == [2, 0]
  rows = (program._workspace["contact_row_data"]
          .reshape(2, program.descriptor.ncontacts_max, 6, 6)[:, :, 0, 0]
          .detach().cpu().numpy())
  assert not np.any(rows[0])
  assert np.count_nonzero(rows[1]) == len(_source_contacts(model)[1])

  # A fresh valid pose restores both worlds and clears the per-world error.
  valid = sim._smooth.run_device(
      state._qpos, state._qvel, state._mpos, state._mquat)["poses"]
  program.generate_candidates(valid, state._qvel)
  assert program._workspace["out_status"].cpu().numpy().tolist() == [0, 0]
  rows = (program._workspace["contact_row_data"]
          .reshape(2, program.descriptor.ncontacts_max, 6, 6)[:, :, 0, 0]
          .detach().cpu().numpy())
  assert all(np.count_nonzero(row) == len(_source_contacts(model)[1])
             for row in rows)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in failed-world step acceptance and replay")
def test_mesh_heightfield_bad_world_step_is_rejected_and_checkpoint_replays():
  model = _model(condim=1, cone="elliptic")
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1",
                        limits=_LIMITS)
  initial_qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2,
                           axis=0)
  sim.reset(qpos=initial_qpos,
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  checkpoint = sim.snapshot()
  initial_acc = sim.state._qacc.detach().cpu().numpy().copy()
  mesh_geom = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  program = sim._coupled_constraints
  generate = program.generate_candidates

  def inject_bad_world(poses, qvel, eq_active=None, **kwargs):
    modified = dict(poses)
    modified["geom_quat"] = poses["geom_quat"].clone()
    # This is after FK and immediately before the actual candidate kernel;
    # body position remains finite, so broadphase still keeps the mesh pair.
    modified["geom_quat"][0, mesh_geom, 0] = float("nan")
    return generate(modified, qvel, eq_active, **kwargs)

  program.generate_candidates = inject_bad_world
  try:
    failed = sim.step(1)
  finally:
    program.generate_candidates = generate
  assert failed.cpu().numpy().tolist() == [2, 0]

  # World zero did not accept the failed candidate, while world one completed
  # the same physical step as the pinned source.
  failed_qpos = sim.state._qpos.detach().cpu().numpy().copy()
  failed_qvel = sim.state._qvel.detach().cpu().numpy().copy()
  failed_time = sim.state._time.detach().cpu().numpy().copy()
  failed_acc = sim.state._qacc.detach().cpu().numpy().copy()
  np.testing.assert_array_equal(failed_qpos[0], initial_qpos[0])
  np.testing.assert_array_equal(failed_qvel[0], np.zeros(model.nv, np.float32))
  np.testing.assert_array_equal(failed_acc[0], initial_acc[0])
  np.testing.assert_array_equal(failed_time[0], 0.0)
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = initial_qpos[1]
  cpu.qvel[:] = 0.0
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(failed_qpos[1], cpu.qpos, rtol=2e-5, atol=3e-5)
  np.testing.assert_allclose(failed_qvel[1], cpu.qvel, rtol=2e-4, atol=5e-4)
  np.testing.assert_allclose(failed_acc[1], cpu.qacc, rtol=2e-3, atol=3e-2)
  assert failed_time[1] == pytest.approx(float(cpu.time), abs=1e-8)

  sim.restore(checkpoint)
  replay_status = sim.step(1)
  assert replay_status.cpu().numpy().tolist() == [0, 0]
  for world in range(2):
    replay_cpu = mujoco.MjData(model)
    replay_cpu.qpos[:] = initial_qpos[world]
    replay_cpu.qvel[:] = 0.0
    mujoco.mj_step(model, replay_cpu)
    np.testing.assert_allclose(sim.state._qpos[world].cpu().numpy(),
                               replay_cpu.qpos, rtol=2e-5, atol=3e-5)
    np.testing.assert_allclose(sim.state._qvel[world].cpu().numpy(),
                               replay_cpu.qvel, rtol=2e-4, atol=5e-4)


def _decode_common_gjk_epa_trace(program):
  layout = program._common_ccd_layout
  assert layout is not None
  raw = program._constants["mesh_hull"].detach().cpu().numpy()
  begin = int(layout["diagnostic_base"])
  count = int(layout["diagnostic_words"])
  trace = raw[begin:begin + count].copy()
  return trace


def _compiled_mesh_vertices(model, mesh_id):
  """Read compiled float vertices using MuJoCo's actual (nvert,3) ABI."""
  points = np.asarray(model.mesh_vert, dtype=np.float32)
  if points.ndim == 1:
    points = points.reshape(-1, 3)
  if points.ndim != 2 or points.shape[1] != 3:
    raise AssertionError(f"unexpected mesh_vert shape {points.shape}")
  address = int(model.mesh_vertadr[mesh_id])
  count = int(model.mesh_vertnum[mesh_id])
  result = np.ascontiguousarray(points[address:address + count])
  assert result.shape == (count, 3)
  return result


def test_common_gjk_epa_trace_record_cpu_layout_and_support_decoder():
  """Exercise the fixed diagnostic record decoder on a synthetic record."""
  from mujoco_metal.common_ccd_bridge import production_ccd_layout

  layout = production_ccd_layout(1, 1, 100)
  assert layout["diagnostic_words"] == 7562
  assert layout["diagnostic_epa_support_offset"] == 2600
  assert layout["diagnostic_epa_face_offset"] == 5032
  assert layout["diagnostic_epa_iteration_offset"] == 5152
  support_offset = 32
  support_words = 38
  intersection_offset = 32 + 48 * support_words
  intersection_stride = 114
  intersection_iterations = 5
  intersection_capacity_words = intersection_iterations * intersection_stride
  intersection_end = intersection_offset + intersection_capacity_words
  marker_offset = support_offset + 63 * support_words
  # Each record reserves three existing support slots. State occupies 38
  # words; the candidate occupies 66 words at offset 38, through offset 103.
  assert 38 + 66 == 104 <= intersection_stride
  assert intersection_offset + intersection_stride == (
      support_offset + 51 * support_words)
  assert intersection_end == marker_offset
  assert marker_offset + support_words < layout["diagnostic_epa_support_offset"]
  assert intersection_end < layout["diagnostic_epa_support_offset"]
  # Prove the largest candidate store is inside its own record, and each
  # record is disjoint from the next one and the truncation marker.
  for iteration_index in range(intersection_iterations):
    record_start = intersection_offset + iteration_index * intersection_stride
    candidate_end = record_start + 38 + 65 + 1
    assert candidate_end <= record_start + intersection_stride
    if iteration_index + 1 < intersection_iterations:
      assert candidate_end <= record_start + intersection_stride
  # A sentinel-backed CPU layout witness confirms the five maximum candidate
  # writes cannot overwrite the next iteration or marker slot.
  arena = np.full(intersection_capacity_words, -1, dtype=np.int32)
  for iteration_index in range(intersection_iterations):
    local_start = iteration_index * intersection_stride
    arena[local_start + 38:local_start + 104] = iteration_index
  assert arena.reshape(intersection_iterations, intersection_stride)[:, 103].tolist() == [
      0, 1, 2, 3, 4]
  assert arena.reshape(intersection_iterations, intersection_stride)[:, 104:].tolist() == [
      [-1] * 10 for _ in range(intersection_iterations)]
  assert marker_offset == intersection_end
  assert marker_offset + support_words <= layout["diagnostic_epa_support_offset"]
  assert intersection_offset >= support_offset + 4 * support_words
  epa_offset = support_offset + 64 * support_words + 116
  assert epa_offset + 20 <= layout["diagnostic_epa_support_offset"]
  assert layout["diagnostic_epa_counts_offset"] == 7328
  assert layout["diagnostic_production_context_offset"] == 7330
  assert layout["diagnostic_production_context_words"] == 232
  assert layout["diagnostic_words"] == (
      layout["diagnostic_production_context_offset"]
      + layout["diagnostic_production_context_words"])
  synthetic = np.zeros(layout["diagnostic_words"], dtype=np.float32)
  synthetic[0:9] = [1, 2, 3, 0, 0, 0, 4, 1, 1]
  synthetic[support_offset:support_offset + support_words] = np.arange(
      support_words, dtype=np.float32)
  synthetic[epa_offset:epa_offset + 20] = np.arange(20, dtype=np.float32)
  synthetic[layout["diagnostic_epa_counts_offset"]:
            layout["diagnostic_epa_counts_offset"] + 2] = [5, 11]
  epa_support_offset = layout["diagnostic_epa_support_offset"]
  synthetic[epa_support_offset:epa_support_offset + support_words] = np.arange(
      100, 100 + support_words, dtype=np.float32)
  epa_face_offset = layout["diagnostic_epa_face_offset"]
  init_face_words = layout["diagnostic_epa_init_face_words"]
  synthetic[epa_face_offset:epa_face_offset + init_face_words] = np.arange(
      200, 200 + init_face_words, dtype=np.float32)
  epa_iteration_offset = layout["diagnostic_epa_iteration_offset"]
  iteration_words = layout["diagnostic_epa_iteration_words"]
  synthetic[epa_iteration_offset:epa_iteration_offset + iteration_words] = np.arange(
      300, 300 + iteration_words, dtype=np.float32)
  assert synthetic[support_offset + 28] == 28
  assert synthetic[support_offset + 29:support_offset + 38].tolist() == [
      29, 30, 31, 32, 33, 34, 35, 36, 37]
  assert synthetic[epa_offset] == 0
  assert synthetic[epa_offset + 4:epa_offset + 7].tolist() == [4, 5, 6]
  # The successful face's six support IDs do not alias its event counts.
  synthetic[epa_offset + 14:epa_offset + 20] = [4, 7, 9, 2, 0, 6]
  assert synthetic[epa_offset + 14:epa_offset + 20].tolist() == [4, 7, 9, 2, 0, 6]
  count_offset = layout["diagnostic_epa_counts_offset"]
  assert synthetic[count_offset:count_offset + 2].tolist() == [5, 11]
  context_offset = layout["diagnostic_production_context_offset"]
  context = synthetic[context_offset:context_offset +
                      layout["diagnostic_production_context_words"]]
  context[0:10] = [20261004, 0, 0, 1, 2, 2, 5, 1, 9, 9]
  assert context[0:10].astype(np.int32).tolist() == [
      20261004, 0, 0, 1, 2, 2, 5, 1, 9, 9]
  assert synthetic[epa_support_offset + 27:epa_support_offset + 29].tolist() == [127, 128]
  assert synthetic[epa_support_offset + 29:epa_support_offset + 38].tolist() == [
      129, 130, 131, 132, 133, 134, 135, 136, 137]
  assert synthetic[epa_face_offset:epa_face_offset + 7].tolist() == list(
      range(200, 207))
  assert synthetic[epa_face_offset + 10:epa_face_offset + 19].tolist() == list(
      range(210, 219))
  assert synthetic[epa_iteration_offset:epa_iteration_offset + 8].tolist() == list(
      range(300, 308))
  assert synthetic[epa_iteration_offset + 32:epa_iteration_offset + 34].tolist() == [
      332, 333]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in bounded common HField GJK/EPA diagnostic")
def test_mesh_heightfield_first_divergence_gjk_epa_support_trace_gpu():
  """Capture the exact chosen prism's GJK supports and EPA witness.

  The trace is written to a fixed, capacity-accounted sidecar in otherwise
  unused workspace. The common candidate rows and solver inputs are unchanged.
  """
  import torch

  model = _model(condim=1, cone="elliptic")
  cpu_data, cpu_contacts = _source_contacts(model)
  # MjContact entries are views owned by this MjData; retain it through the trace.
  assert cpu_data.ncon == len(cpu_contacts)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1",
                        limits=_LIMITS)
  program = sim._coupled_constraints
  layout = program._common_ccd_layout
  assert layout is not None and layout["diagnostic_words"] == 7562
  info = program._constants["mesh_hull_info"]
  header = int(layout["header_base"])
  assert int(info[header + 29].item()) == -1
  control = sim.assembled_system(recompute=True)
  control_outputs = {key: control[key].clone() for key in (
      "contact_mask", "contact_distance", "contact_position", "contact_normal",
      "J", "R", "ar", "W", "qacc", "qfrc_constraint", "status")}
  trace_selector = torch.tensor([0, 2, 5, 1], dtype=torch.int32,
                                device=info.device)
  info[header + 29:header + 33].copy_(trace_selector)
  assembly = sim.assembled_system(recompute=True)
  for key, expected in control_outputs.items():
    torch.testing.assert_close(assembly[key], expected, rtol=0, atol=0,
                               msg=f"observer changed public {key}")
  trace = _decode_common_gjk_epa_trace(program)
  assert int(trace[0]) == 1
  assert trace[1:4].astype(np.int32).tolist() == [2, 5, 1]
  support_count = int(trace[8])
  assert support_count > 0
  assert int(trace[31]) == int(support_count > 64)
  support_capacity = 64
  print("HFIELD_TRACE_QUERY_ROW_COL_TRI", trace[:4].tolist())
  print("HFIELD_TRACE_GJK_STATUS_ITERS_COUNT_NEEDS_SUPPORTS_TRUNCATED",
        [*trace[4:9].astype(np.int32).tolist(), int(trace[31])])
  print("HFIELD_TRACE_GJK_DISTANCE", trace[9:12].tolist())
  print("HFIELD_TRACE_GJK_WITNESS_A", trace[12:21].tolist())
  print("HFIELD_TRACE_GJK_WITNESS_B", trace[21:30].tolist())
  support_base = 32
  support_words = 38
  support_sequence = trace[
      support_base:support_base + min(support_count, support_capacity) * support_words
  ].reshape(-1, support_words)
  print("HFIELD_TRACE_SUPPORT_SEQUENCE", support_sequence[:, :29].tolist())
  directions = support_sequence[:, 29:38].reshape(-1, 3, 3).sum(axis=2)
  print("HFIELD_TRACE_SUPPORT_DIRECTIONS", directions.tolist())
  intersection_count = int(trace[30])
  intersection_base = 32 + 48 * support_words
  intersection_stride = 3 * support_words
  assert 0 <= intersection_count <= 5
  intersection_trace = trace[
      intersection_base:intersection_base
      + intersection_count * intersection_stride
  ].reshape(-1, intersection_stride)
  intersection_truncated = bool(
      trace[32 + 63 * support_words + support_words - 1])
  print("HFIELD_TRACE_INTERSECTION_COUNT", intersection_count)
  print("HFIELD_TRACE_INTERSECTION_TRUNCATED", intersection_truncated)
  decoded_intersection = []
  for record in intersection_trace:
    assert record.shape == (intersection_stride,)
    state, candidate = record[:support_words], record[38:104]
    assert state.shape == (38,) and candidate.shape == (66,)
    decoded_intersection.append({
        "iteration": int(state[0]),
        "order": state[1:5].astype(np.int32).tolist(),
        "face_distances_hlt": state[5:17].reshape(4, 3).tolist(),
        "selected_face": int(state[17]),
        "selected_normal_hlt": state[18:27].tolist(),
        "simplex_feature_ids": list(zip(
            state[27:31].astype(np.int32).tolist(),
            state[31:35].astype(np.int32).tolist())),
        "ordered_simplex_minkowski_hlt": candidate[:36].reshape(4, 9).tolist(),
        "candidate_minkowski_hlt": candidate[36:45].tolist(),
        "candidate_point_a_hlt": candidate[45:54].tolist(),
        "candidate_point_b_hlt": candidate[54:63].tolist(),
        "candidate_feature_ids": candidate[63:65].astype(np.int32).tolist(),
        "support_status": int(candidate[65]),
    })
  print("HFIELD_TRACE_INTERSECTION_RECORDS", decoded_intersection)
  from mujoco_metal.common_ccd_bridge import (
      compiled_mesh_source_support_index, hfield_prism_support_record,
      hfield_source_support_index)
  terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  _, prism_words, prism_high = hfield_prism_support_record(
      model, terrain, *trace[1:4].astype(np.int32).tolist(),
      0.0, vertex_offset=0)
  prism_residual = prism_words[48:84].reshape(6, 3, 2)
  prism_vertices = prism_high.reshape(6, 3).astype(np.float64)
  prism_vertices += prism_residual[:, :, 0].astype(np.float64)
  prism_vertices += prism_residual[:, :, 1].astype(np.float64)
  source_ids = [hfield_source_support_index(prism_vertices, direction)
                for direction in directions]
  candidate_ids = support_sequence[:, 27].astype(np.int32).tolist()
  print("HFIELD_TRACE_PINNED_SUPPORT_IDS", source_ids)
  print("HFIELD_TRACE_CANDIDATE_SUPPORT_IDS", candidate_ids)
  print("HFIELD_TRACE_PRISM_SUPPORT_ID_MATCH", [
      source_id == candidate_id for source_id, candidate_id
      in zip(source_ids, candidate_ids)])
  mesh_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  mesh_id = int(model.geom_dataid[mesh_geom])
  mesh_vertices = _compiled_mesh_vertices(model, mesh_id)
  mesh_world_to_local = np.asarray(
      cpu_data.geom_xmat[mesh_geom], dtype=np.float64).reshape(3, 3).T
  mesh_cache = -1
  for axis in np.eye(3):
    for sign in (1.0, -1.0):
      mesh_cache = compiled_mesh_source_support_index(
          mesh_vertices, sign * axis, mesh_cache)
  source_mesh_ids = []
  for direction in directions:
    local_direction = mesh_world_to_local @ (-direction)
    mesh_cache = compiled_mesh_source_support_index(
        mesh_vertices, local_direction, mesh_cache)
    source_mesh_ids.append(mesh_cache)
  candidate_mesh_ids = support_sequence[:, 28].astype(np.int32).tolist()
  print("HFIELD_TRACE_PINNED_MESH_IDS", source_mesh_ids)
  print("HFIELD_TRACE_CANDIDATE_MESH_IDS", candidate_mesh_ids)
  print("HFIELD_TRACE_MESH_ID_MATCH", [
      source_id == candidate_id for source_id, candidate_id
      in zip(source_mesh_ids, candidate_mesh_ids)])
  simplex_base = support_base + support_capacity * support_words
  simplex = trace[simplex_base:simplex_base + 4 * 29].reshape(4, 29)
  print("HFIELD_TRACE_FINAL_SIMPLEX", simplex.tolist())
  epa_base = simplex_base + 4 * 29
  print("HFIELD_TRACE_EPA_STATUS_DISTANCE_FACE_CONTACT",
        trace[epa_base:epa_base + 20].tolist())
  count_offset = layout["diagnostic_epa_counts_offset"]
  epa_support_count = int(trace[count_offset])
  epa_face_count = int(trace[count_offset + 1])
  epa_support_capacity = layout["diagnostic_epa_support_capacity"]
  epa_support_base = layout["diagnostic_epa_support_offset"]
  epa_support = trace[
      epa_support_base:epa_support_base + min(
          epa_support_count, epa_support_capacity) * support_words
  ].reshape(-1, support_words)
  print("HFIELD_TRACE_EPA_SUPPORT_COUNT_CAPACITY",
        [epa_support_count, epa_support_capacity])
  print("HFIELD_TRACE_EPA_SUPPORT_SEQUENCE", epa_support.tolist())
  epa_face_base = layout["diagnostic_epa_face_offset"]
  initial_faces = min(epa_face_count, 6)
  initial_face_words = layout["diagnostic_epa_init_face_words"]
  init_faces = trace[
      epa_face_base:epa_face_base + initial_faces * initial_face_words
  ].reshape(-1, initial_face_words)
  print("HFIELD_TRACE_EPA_INITIAL_FACES", init_faces.tolist())
  epa_iterations = max(0, epa_face_count - initial_faces)
  iter_capacity = layout["diagnostic_epa_iteration_capacity"]
  iter_base = layout["diagnostic_epa_iteration_offset"]
  iter_words = layout["diagnostic_epa_iteration_words"]
  iterations = trace[
      iter_base:iter_base + min(epa_iterations, iter_capacity) * iter_words
  ].reshape(-1, iter_words)
  print("HFIELD_TRACE_EPA_ITERATION_COUNT_CAPACITY",
        [epa_iterations, iter_capacity])
  print("HFIELD_TRACE_EPA_SELECTED_FACE_SEQUENCE", iterations.tolist())
  context_base = layout["diagnostic_production_context_offset"]
  context = trace[context_base:context_base +
                  layout["diagnostic_production_context_words"]]
  assert int(context[0]) == 20261004
  assert context[5:8].astype(np.int32).tolist() == [2, 5, 1]
  context_fields = {
      "metadata": context[0:10].tolist(),
      "geom_pos_hfield_mesh": context[10:16].reshape(2, 3).tolist(),
      "geom_quat_hfield_mesh": context[16:24].reshape(2, 4).tolist(),
      "geom_rotation_hfield_mesh": context[24:42].reshape(2, 3, 3).tolist(),
      "hfield_size_high_mid_tail": context[42:54].reshape(4, 3).tolist(),
      "grid_dx_dy_high_mid_tail": context[54:60].reshape(2, 3).tolist(),
      "local_pos_high_mid_tail": context[63:72].reshape(3, 3).tolist(),
      "local_matrix_high_mid_tail": context[72:99].reshape(3, 3, 3).tolist(),
      "compiled_mesh_center_high_mid_tail": context[99:108].reshape(3, 3).tolist(),
      "compiled_mesh_matrix_high_mid_tail": context[108:135].reshape(9, 3).tolist(),
      "compiled_mesh_size_high_mid_tail": context[135:144].reshape(3, 3).tolist(),
      "compiled_mesh_margin_high_mid_tail": context[144:147].tolist(),
      "prism_center_high_mid_tail": context[159:168].reshape(3, 3).tolist(),
      "mesh_center_high_mid_tail": context[168:177].reshape(3, 3).tolist(),
      "prism_vertices_high_mid_tail": context[177:231].reshape(6, 3, 3).tolist(),
  }
  # Matrices are packed scalar words; they must not overlap size records.
  # Compare the actual observer quaternion and matrix with an independent
  # upstream quaternion conversion, rather than a shader-layout assertion.
  for index, quat in enumerate(context_fields["geom_quat_hfield_mesh"]):
    expected_matrix = np.empty(9, np.float64)
    mujoco.mju_quat2Mat(expected_matrix, np.asarray(quat, np.float64))
    np.testing.assert_allclose(
        context_fields["geom_rotation_hfield_mesh"][index],
        expected_matrix.reshape(3, 3), rtol=0, atol=2e-7)
  print("HFIELD_TRACE_PRODUCTION_POSE_AND_CENTER_OPERANDS", context_fields)
  represented_data = mujoco.MjData(model)
  represented_data.qpos[:] = sim.state.qpos[0].detach().cpu().numpy().astype(np.float64)
  represented_data.qvel[:] = sim.state.qvel[0].detach().cpu().numpy().astype(np.float64)
  mujoco.mj_forward(model, represented_data)
  terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  mesh_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  print("HFIELD_TRACE_SOURCE_QPOS0", np.asarray(model.qpos0).tolist())
  print("HFIELD_TRACE_PUBLIC_QPOS_HIGH",
        sim.state.qpos[0].detach().cpu().numpy().tolist())
  print("HFIELD_TRACE_SOURCE_FULL_POSE",
        {"hfield_pos": cpu_data.geom_xpos[terrain].tolist(),
         "mesh_pos": cpu_data.geom_xpos[mesh_geom].tolist(),
         "hfield_mat": cpu_data.geom_xmat[terrain].reshape(3, 3).tolist(),
         "mesh_mat": cpu_data.geom_xmat[mesh_geom].reshape(3, 3).tolist()})
  print("HFIELD_TRACE_SOURCE_REPRESENTED_POSE",
        {"hfield_pos": represented_data.geom_xpos[terrain].tolist(),
         "mesh_pos": represented_data.geom_xpos[mesh_geom].tolist(),
         "hfield_mat": represented_data.geom_xmat[terrain].reshape(3, 3).tolist(),
         "mesh_mat": represented_data.geom_xmat[mesh_geom].reshape(3, 3).tolist()})
  raw_ids = _raw_common_prism_ids(program)
  selected = np.flatnonzero(np.all(raw_ids == np.asarray([2, 5, 1]), axis=1))
  assert selected.tolist() == [5]
  print("HFIELD_TRACE_ORDERED_CPU_CONTACT",
        int(selected[0]), float(cpu_contacts[int(selected[0])].dist),
        np.asarray(cpu_contacts[int(selected[0])].pos).tolist(),
        np.asarray(cpu_contacts[int(selected[0])].frame[:3]).tolist())
  # Ensure diagnostics did not alter the production outputs.
  status, produced = _raw_common_contacts(program)
  consumed = _native_contacts(assembly)
  assert status == 0 and len(produced) == len(consumed) == len(cpu_contacts)
  _assert_raw_common_matches_rows(produced, consumed)


def test_mesh_support_oracle_uses_world_to_local_rotation_and_vertex_rows():
  """The GJK operand is world-space; mjc_meshSupport consumes local-space."""
  from mujoco_metal.common_ccd_bridge import compiled_mesh_source_support_index

  xml = _XML.replace('<body name="mesh_body" pos="0 0 .03">',
                     '<body name="mesh_body" pos="0 0 .03" euler=".21 -.17 .31">')
  model = _model(xml=xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  mesh_id = int(model.geom_dataid[geom])
  vertices = _compiled_mesh_vertices(model, mesh_id)
  assert vertices.shape == (int(model.mesh_vertnum[mesh_id]), 3)
  world_to_local = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3).T
  direction = np.asarray([0.37, -0.62, 0.81], dtype=np.float64)
  expected = compiled_mesh_source_support_index(vertices, world_to_local @ direction)
  # Rotating the compiled points to world space and dotting the untransformed
  # operand is the equivalent independent frame check.
  local_to_world = world_to_local.T
  world_vertices = vertices.astype(np.float64) @ local_to_world.T
  scores = world_vertices @ direction
  assert expected == int(np.argmax(scores))


def test_hfield_source_contact_views_remain_owned_through_trace_cpu():
  """Retain the actual MjData while helper scratch outputs replace temporary names."""
  import gc
  from mujoco_metal.common_ccd_bridge import hfield_prism_support_record
  model = _model(condim=1, cone="elliptic")
  cpu_data, contacts = _source_contacts(model)
  expected = [(float(c.dist), np.array(c.pos, copy=True),
               np.array(c.frame, copy=True)) for c in contacts]
  terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  _, prism_words, prism_high = hfield_prism_support_record(
      model, terrain, 2, 3, 0, 0.0, vertex_offset=0)
  gc.collect()
  assert cpu_data.ncon == len(contacts) == len(expected)
  assert prism_words.size > 0 and prism_high.size > 0
  for contact, (distance, position, frame) in zip(contacts, expected):
    assert float(contact.dist) == distance
    np.testing.assert_array_equal(contact.pos, position)
    np.testing.assert_array_equal(contact.frame, frame)
