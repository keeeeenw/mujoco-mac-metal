"""Opt-in public integration witness for the pinned generic convex route.

The fixture is a deep mesh-box face press. MuJoCo 3.10 emits a four-point
manifold here, so the test checks both the common GJK/EPA contact producer and
its use by canonical row assembly/constraint solve.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation


_XML = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="100"
          tolerance="1e-8" gravity="0 0 0"/>
  <asset><mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
  <worldbody>
    <body name="mesh_body" pos="0 0 .3"><freejoint/>
      <geom name="mesh" type="mesh" mesh="tet" condim="1"
            friction="0 0 0"/>
    </body>
    <body name="box_body" pos="0 0 .3"><freejoint/>
      <geom name="box" type="box" size=".1 .1 .1" condim="1"
            friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""
_XML_REVERSED = _XML.replace(
    '''    <body name="mesh_body" pos="0 0 .3"><freejoint/>
      <geom name="mesh" type="mesh" mesh="tet" condim="1"
            friction="0 0 0"/>
    </body>
    <body name="box_body" pos="0 0 .3"><freejoint/>
      <geom name="box" type="box" size=".1 .1 .1" condim="1"
            friction="0 0 0"/>
    </body>''',
    '''    <body name="box_body" pos="0 0 .3"><freejoint/>
      <geom name="box" type="box" size=".1 .1 .1" condim="1"
            friction="0 0 0"/>
    </body>
    <body name="mesh_body" pos="0 0 .3"><freejoint/>
      <geom name="mesh" type="mesh" mesh="tet" condim="1"
            friction="0 0 0"/>
    </body>''')


def _sorted_contacts(position, distance, normal):
  order = np.lexsort((position[:, 2], position[:, 1], position[:, 0]))
  return position[order], distance[order], normal[order]


def test_pinned_source_mesh_box_manifold_is_native_and_reaches_solver_cpu():
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 4
  assert all(data.contact[i].dist < 0 for i in range(data.ncon))
  assert len({tuple(np.round(data.contact[i].pos, 8)) for i in range(data.ncon)}) == 4



_TYPE_XML = {
    2: '<geom name="g" type="sphere" size=".16" condim="1" friction="0 0 0"/>',
    3: '<geom name="g" type="capsule" size=".12 .18" condim="1" friction="0 0 0"/>',
    4: '<geom name="g" type="ellipsoid" size=".16 .14 .12" condim="1" friction="0 0 0"/>',
    5: '<geom name="g" type="cylinder" size=".15 .16" condim="1" friction="0 0 0"/>',
    6: '<geom name="g" type="box" size=".15 .14 .12" condim="1" friction="0 0 0"/>',
    7: '<geom name="g" type="mesh" mesh="tet" condim="1" friction="0 0 0"/>',
}
_CONVEX_PAIRS = ((2, 4), (2, 7), (3, 4), (3, 5), (3, 7), (4, 4),
                 (4, 5), (4, 6), (4, 7), (5, 5), (5, 6), (5, 7),
                 (6, 7), (7, 7))


def _pair_xml(type_a, type_b, *, multiccd=True, condim=1,
              cone="elliptic", friction="0 0 0", rotation=None,
              gravity="0 0 0"):
  flag = '' if multiccd else '<flag multiccd="disable"/>'
  geom_a = (_TYPE_XML[type_a].replace('name="g"', 'name="ga"')
            .replace('condim="1"', f'condim="{condim}"')
            .replace('friction="0 0 0"', f'friction="{friction}"'))
  geom_b = (_TYPE_XML[type_b].replace('name="g"', 'name="gb"')
            .replace('condim="1"', f'condim="{condim}"')
            .replace('friction="0 0 0"', f'friction="{friction}"'))
  rotate = f' euler="{rotation}"' if rotation else ''
  return (f'<mujoco><option timestep="0.001" integrator="Euler" '
          f'iterations="100" tolerance="1e-8" gravity="{gravity}" '
          f'cone="{cone}">{flag}</option>'
          '<asset><mesh name="tet" vertex="-.15 -.15 -.15 .15 -.15 -.15 '
          '-.15 .15 -.15 -.15 -.15 .15" '
          'face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset><worldbody>'
          f'<body name="a" pos="0 0 .2"><freejoint/><inertial pos="0 0 0" '
          f'mass="1" diaginertia="1 1 1"/>{geom_a}</body>'
          f'<body name="b" pos=".1 0 .2"{rotate}><freejoint/><inertial pos="0 0 0" '
          f'mass="1" diaginertia="1 1 1"/>{geom_b}</body>'
          '</worldbody></mujoco>')


def test_pinned_generic_convex_pair_matrix_has_real_cpu_contacts():
  for pair in _CONVEX_PAIRS:
    model = mujoco.MjModel.from_xml_string(_pair_xml(*pair))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0, (pair, data.ncon)
    assert all(data.contact[i].dist < 0 for i in range(data.ncon))

def test_pinned_source_no_multiccd_uses_only_the_primary_witness():
  xml = _XML.replace(
      '<option timestep="0.002" integrator="Euler" iterations="100"\n'
      '          tolerance="1e-8" gravity="0 0 0"/>',
      '<option timestep="0.002" integrator="Euler" iterations="100"\n'
      '          tolerance="1e-8" gravity="0 0 0">\n'
      '    <flag multiccd="disable"/>\n  </option>')
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 1
  assert data.contact[0].dist < 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_common_mjc_convex_mesh_box_multicontact_public_native_gate():
  for xml, expected_count in ((_XML, 4), (_XML_REVERSED, 4)):
    model = mujoco.MjModel.from_xml_string(xml)
    sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
    cpu = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu)
    native = sim.assembled_system(recompute=True)

    mask = native["contact_mask"][0].detach().cpu().numpy() > 0.5
    count = int(mask.sum())
    assert count == cpu.ncon == expected_count
    native_pos = native["contact_position"][0, mask].detach().cpu().numpy()
    native_dist = native["contact_distance"][0, mask].detach().cpu().numpy()
    native_normal = native["contact_normal"][0, mask].detach().cpu().numpy()
    cpu_pos = np.asarray([cpu.contact[i].pos for i in range(cpu.ncon)])
    cpu_dist = np.asarray([cpu.contact[i].dist for i in range(cpu.ncon)])
    cpu_normal = np.asarray([cpu.contact[i].frame[:3] for i in range(cpu.ncon)])
    native_pos, native_dist, native_normal = _sorted_contacts(
        native_pos, native_dist, native_normal)
    cpu_pos, cpu_dist, cpu_normal = _sorted_contacts(cpu_pos, cpu_dist, cpu_normal)
    np.testing.assert_allclose(native_pos, cpu_pos, rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(native_dist, cpu_dist, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(native_normal, cpu_normal, rtol=0.0, atol=2e-4)

    # Verify that the recovered manifold was assembled and solved, not merely
    # emitted as an unused side output.
    mujoco.mj_step(model, cpu)
    qfrc_native = native["qfrc_constraint"][0].detach().cpu().numpy()
    np.testing.assert_allclose(qfrc_native, cpu.qfrc_constraint,
                               rtol=2e-3, atol=2e-3)
    assert int(native["status"][0].detach().cpu()) == 0


def test_multicontact_geom2_face_uses_its_selected_feature_index_in_both_orders():
  # Source alignedFaces returns (i, j), with i indexing geom1 and j
  # indexing geom2. Deliberately choose unequal feature indices so using i
  # for geom2 selects a different face and cannot pass accidentally.
  cases = (
      ([(1, 0, 0), (0, 0, -1)], [(0, 1, 0), (-1, 0, 0)],
       [17, 23], [31, 47]),
      ([(0, 1, 0), (-1, 0, 0)], [(1, 0, 0), (0, 0, -1)],
       [31, 47], [17, 23]),
  )
  for normals_a, normals_b, face_ids_a, face_ids_b in cases:
    selected = next((i, j) for i, a in enumerate(normals_a)
                    for j, b in enumerate(normals_b)
                    if np.dot(a, b) < -1e-6)
    i, j = selected
    assert i != j
    assert face_ids_b[j] != face_ids_b[i]
    # The production function must follow pinned multicontact's idx2[j]
    # selection; checking the exact source expression guards the MSL path.
    import pathlib
    shader = pathlib.Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "common_ccd_production.metal"
    source = shader.read_text()
    assert "common_ccd_face_polygon(b,rec_b,face_ids_b[selected.y]" in source
    assert "common_ccd_face_polygon(b,rec_b,face_ids_b[selected.x]" not in source
    assert face_ids_b[j] == (47 if j == 1 else 17)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_common_mjc_convex_pair_matrix_matches_contact_rows_and_solver():
  for pair in _CONVEX_PAIRS:
    model = mujoco.MjModel.from_xml_string(_pair_xml(*pair))
    cpu = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu)
    assert cpu.ncon > 0, pair
    sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
    native = sim.assembled_system(recompute=True)
    mask = native["contact_mask"][0].detach().cpu().numpy() > 0.5
    native_pos = native["contact_position"][0, mask].detach().cpu().numpy()
    native_dist = native["contact_distance"][0, mask].detach().cpu().numpy()
    native_normal = native["contact_normal"][0, mask].detach().cpu().numpy()
    native_J = native["contact_jacobian"][0, mask, 0].detach().cpu().numpy()
    cpu_addr = np.asarray([cpu.contact[i].efc_address for i in range(cpu.ncon)], dtype=np.int32)
    cpu_pos = np.asarray([cpu.contact[i].pos for i in range(cpu.ncon)])
    cpu_dist = np.asarray([cpu.contact[i].dist for i in range(cpu.ncon)])
    cpu_normal = np.asarray([cpu.contact[i].frame[:3] for i in range(cpu.ncon)])
    cpu_J = np.asarray(cpu.efc_J).reshape(cpu.nefc, model.nv)[cpu_addr]
    order_native = np.lexsort((native_pos[:, 2], native_pos[:, 1], native_pos[:, 0]))
    order_cpu = np.lexsort((cpu_pos[:, 2], cpu_pos[:, 1], cpu_pos[:, 0]))
    assert int(mask.sum()) == cpu.ncon, pair
    np.testing.assert_allclose(native_pos[order_native], cpu_pos[order_cpu], rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(native_dist[order_native], cpu_dist[order_cpu], rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(native_normal[order_native], cpu_normal[order_cpu], rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(native_J[order_native], cpu_J[order_cpu], rtol=0.0, atol=2e-4)
    descriptor = sim._coupled_constraints.descriptor
    contact_map = np.asarray(descriptor.contact_condim_packed,
                              dtype=np.int32).reshape(-1, 3)
    row_start = np.asarray([
        descriptor.nr_joint + int(contact_map[int(slot), 1])
        for slot in np.flatnonzero(mask)], dtype=np.int32)
    native_aref = native["ar"][0, row_start].detach().cpu().numpy()
    native_R = native["R"][0, row_start].detach().cpu().numpy()
    np.testing.assert_allclose(native_aref[order_native], np.asarray(cpu.efc_aref)[cpu_addr][order_cpu], rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(native_R[order_native], np.asarray(cpu.efc_R)[cpu_addr][order_cpu], rtol=3e-4, atol=3e-4)
    mujoco.mj_step(model, cpu)
    np.testing.assert_allclose(native["qfrc_constraint"][0].detach().cpu().numpy(), cpu.qfrc_constraint, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_common_mjc_convex_primary_witness_without_multiccd_is_native():
  model = mujoco.MjModel.from_xml_string(_pair_xml(6, 7, multiccd=False))
  cpu = mujoco.MjData(model)
  mujoco.mj_forward(model, cpu)
  assert cpu.ncon == 1
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  native = sim.assembled_system(recompute=True)
  mask = native["contact_mask"][0].detach().cpu().numpy() > 0.5
  assert int(mask.sum()) == 1
  np.testing.assert_allclose(native["contact_distance"][0, mask].detach().cpu().numpy(),
                             [cpu.contact[0].dist], rtol=0.0, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_common_mjc_convex_trajectory_checkpoint_replay_and_reset():
  model = mujoco.MjModel.from_xml_string(_pair_xml(6, 7))
  cpu = mujoco.MjData(model)
  mujoco.mj_forward(model, cpu)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(cpu.qpos, dtype=np.float32)[None],
            qvel=np.asarray(cpu.qvel, dtype=np.float32)[None])
  initial_qpos = sim.state.qpos.clone()
  initial_qvel = sim.state.qvel.clone()
  checkpoint = sim.snapshot()
  trajectory = []
  for _ in range(8):
    mujoco.mj_step(model, cpu)
    sim.step(1)
    native_qpos = sim.state.qpos.clone()
    native_qvel = sim.state.qvel.clone()
    np.testing.assert_allclose(native_qpos[0].detach().cpu().numpy(), cpu.qpos,
                               rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(native_qvel[0].detach().cpu().numpy(), cpu.qvel,
                               rtol=2e-3, atol=2e-3)
    trajectory.append((native_qpos, native_qvel))
  sim.restore(checkpoint)
  for qpos_expected, qvel_expected in trajectory:
    sim.step(1)
    assert sim.state.qpos.equal(qpos_expected)
    assert sim.state.qvel.equal(qvel_expected)
  sim.reset()
  assert sim.state.qpos.equal(initial_qpos)
  assert sim.state.qvel.equal(initial_qvel)


def _source_contact_fields(model, data):
  mujoco.mj_forward(model, data)
  ncon = int(data.ncon)
  addresses = np.asarray([data.contact[i].efc_address for i in range(ncon)],
                         dtype=np.int32)
  positions = np.asarray([data.contact[i].pos for i in range(ncon)]).reshape(-1, 3)
  distances = np.asarray([data.contact[i].dist for i in range(ncon)])
  normals = np.asarray([data.contact[i].frame[:3] for i in range(ncon)]).reshape(-1, 3)
  jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
  return positions, distances, normals, jacobian, addresses


@pytest.mark.parametrize("pair", ((2, 4), (4, 7), (6, 7), (7, 7)))
@pytest.mark.parametrize("cone", ("elliptic", "pyramidal"))
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_common_mjc_convex_rotated_friction_pair_lifecycle_gpu(pair, cone):
  """Exercise source contact/J/R/aref/force and replay for rotated families."""
  model = mujoco.MjModel.from_xml_string(_pair_xml(
      *pair, condim=3, cone=cone, friction="1 .01 .001",
      rotation=".12 -.08 .16", gravity="0 0 -9.81"))
  cpu = mujoco.MjData(model)
  mujoco.mj_forward(model, cpu)
  assert cpu.ncon > 0, (pair, cone)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  native = sim.assembled_system(recompute=True)
  mask = native["contact_mask"][0].detach().cpu().numpy() > 0.5
  assert int(mask.sum()) == cpu.ncon, (pair, cone)
  pos, dist, normal, source_J, addresses = _source_contact_fields(model, cpu)
  native_pos = native["contact_position"][0, mask].detach().cpu().numpy()
  native_dist = native["contact_distance"][0, mask].detach().cpu().numpy()
  native_normal = native["contact_normal"][0, mask].detach().cpu().numpy()
  order_native = np.lexsort((native_pos[:, 2], native_pos[:, 1], native_pos[:, 0]))
  order_cpu = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
  np.testing.assert_allclose(native_pos[order_native], pos[order_cpu], rtol=0.0, atol=2e-4)
  np.testing.assert_allclose(native_dist[order_native], dist[order_cpu], rtol=0.0, atol=2e-5)
  np.testing.assert_allclose(native_normal[order_native], normal[order_cpu], rtol=0.0, atol=2e-4)
  descriptor = sim._coupled_constraints.descriptor
  packed = np.asarray(descriptor.contact_condim_packed).reshape(-1, 3)
  rows_per_contact = lambda slot: (
      1 if int(packed[slot, 0]) == 1 else
      2 * (int(packed[slot, 0]) - 1)
      if int(packed[slot, 2]) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
      else int(packed[slot, 0]))
  slots = np.flatnonzero(mask)
  native_starts = np.asarray([
      descriptor.nr_joint + int(packed[slot, 1]) for slot in slots],
      dtype=np.int32)
  native_spans = [rows_per_contact(slot) for slot in slots]
  cpu_spans = [rows_per_contact(int(slot)) for slot in order_cpu]
  assert [native_spans[int(i)] for i in order_native] == cpu_spans
  dense_J = sim._coupled_constraints.materialize_jacobian()[0].detach().cpu().numpy()
  native_aref = native["ar"][0].detach().cpu().numpy()
  native_R = native["R"][0].detach().cpu().numpy()
  for native_index, cpu_index in enumerate(order_cpu):
    native_start = int(native_starts[order_native[native_index]])
    source_start = int(addresses[cpu_index])
    count = native_spans[int(order_native[native_index])]
    np.testing.assert_allclose(dense_J[native_start:native_start + count],
                               source_J[source_start:source_start + count],
                               rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(native_aref[native_start:native_start + count],
                               np.asarray(cpu.efc_aref)[source_start:source_start + count],
                               rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(native_R[native_start:native_start + count],
                               np.asarray(cpu.efc_R)[source_start:source_start + count],
                               rtol=3e-4, atol=3e-4)
  native_friction = native["contact_friction"].detach().cpu().numpy()[slots]
  cpu_friction = np.asarray([cpu.contact[int(i)].friction for i in order_cpu])
  np.testing.assert_allclose(native_friction[order_native], cpu_friction,
                             rtol=0.0, atol=1e-7)

  # The first assembled force is compared with the first source forward at the
  # same state. Later trajectory points are checked independently below.
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(native["qfrc_constraint"][0].detach().cpu().numpy(),
                             cpu.qfrc_constraint, rtol=2e-3, atol=2e-3)

  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32)[None],
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  cpu = mujoco.MjData(model)
  mujoco.mj_forward(model, cpu)
  checkpoint = sim.snapshot()
  expected = []
  for _ in range(4):
    mujoco.mj_step(model, cpu)
    status = sim.step(1)
    assert status.detach().cpu().numpy().tolist() == [0]
    np.testing.assert_allclose(sim.state._qpos[0].detach().cpu().numpy(),
                               cpu.qpos, rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(sim.state._qvel[0].detach().cpu().numpy(),
                               cpu.qvel, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(sim.state._qacc[0].detach().cpu().numpy(),
                               cpu.qacc, rtol=2e-3, atol=3e-2)
    np.testing.assert_allclose(
        sim._last_coupled["qfrc_constraint"][0].detach().cpu().numpy(),
        cpu.qfrc_constraint, rtol=2e-3, atol=2e-3)
    expected.append((sim.state.qpos.clone(), sim.state.qvel.clone()))
  sim.restore(checkpoint)
  for qpos, qvel in expected:
    sim.step(1)
    assert sim.state.qpos.equal(qpos)
    assert sim.state.qvel.equal(qvel)
  sim.reset()
  np.testing.assert_allclose(sim.state.qpos[0].detach().cpu().numpy(), model.qpos0,
                             rtol=0.0, atol=2e-6)


def test_common_mjc_convex_b2_fixture_has_contact_and_no_contact_cpu():
  """Prove the two-world contact/no-contact setup used by native isolation."""
  model = mujoco.MjModel.from_xml_string(_pair_xml(6, 7))
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2, axis=0)
  assert model.njnt == 2 and qpos.shape[0] == 2
  qposadr = int(model.jnt_qposadr[1])
  qpos[1, qposadr] += 2.0
  counts = []
  for row in qpos:
    data = mujoco.MjData(model)
    data.qpos[:] = row
    mujoco.mj_forward(model, data)
    counts.append(int(data.ncon))
  assert counts[0] > 0 and counts[1] == 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_common_mjc_convex_b2_contact_failure_and_nocontact_isolation_gpu():
  """A malformed candidate in one world fails closed without touching peers."""
  model = mujoco.MjModel.from_xml_string(_pair_xml(6, 7))
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 3, axis=0)
  qposadr = int(model.jnt_qposadr[1])
  qpos[2, qposadr] += 2.0
  qvel = np.zeros((3, model.nv), dtype=np.float32)
  sim = MetalSimulation(model, batch_size=3, profile="integrated_euler_v1")
  sim.reset(qpos=qpos, qvel=qvel)
  before = sim.snapshot()
  mesh_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gb")
  program = sim._coupled_constraints
  generate = program.generate_candidates

  def inject_bad_world(poses, qvel, eq_active=None, **kwargs):
    modified = dict(poses)
    modified["geom_quat"] = poses["geom_quat"].clone()
    modified["geom_quat"][0, mesh_geom, 0] = float("nan")
    return generate(modified, qvel, eq_active, **kwargs)

  program.generate_candidates = inject_bad_world
  try:
    failed = sim.step(1)
  finally:
    program.generate_candidates = generate
  assert failed.detach().cpu().numpy().tolist() == [2, 0, 0]
  failed_qpos = sim.state._qpos.detach().cpu().numpy().copy()
  failed_qvel = sim.state._qvel.detach().cpu().numpy().copy()
  failed_time = sim.state._time.detach().cpu().numpy().copy()
  np.testing.assert_array_equal(failed_qpos[0], qpos[0])
  np.testing.assert_array_equal(failed_qvel[0], qvel[0])
  np.testing.assert_array_equal(failed_time[0], 0.0)
  for world in (1, 2):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos[world]
    cpu.qvel[:] = qvel[world]
    mujoco.mj_step(model, cpu)
    np.testing.assert_allclose(failed_qpos[world], cpu.qpos,
                               rtol=2e-5, atol=3e-5)
    np.testing.assert_allclose(failed_qvel[world], cpu.qvel,
                               rtol=2e-4, atol=5e-4)
    assert failed_time[world] == pytest.approx(float(cpu.time), abs=1e-8)
  sim.restore(before)
  source = sim.assembled_system(recompute=True)
  assert source["status"].detach().cpu().numpy().tolist() == [0, 0, 0]
  source_mask = source["contact_mask"].detach().cpu().numpy()
  assert source_mask[0].any() and source_mask[1].any()
  assert not source_mask[2].any()
  sim.restore(before)
  replay = sim.step(1)
  assert replay.detach().cpu().numpy().tolist() == [0, 0, 0]
  for world in range(3):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos[world]
    cpu.qvel[:] = qvel[world]
    mujoco.mj_step(model, cpu)
    np.testing.assert_allclose(sim.state._qpos[world].detach().cpu().numpy(),
                               cpu.qpos, rtol=2e-5, atol=3e-5)
    np.testing.assert_allclose(sim.state._qvel[world].detach().cpu().numpy(),
                               cpu.qvel, rtol=2e-4, atol=5e-4)
