# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Strict source-oracle qualification for rigid bundled-plugin SDF contacts.

This is a stronger companion to the descriptor smoke tests: it matches each
pinned MuJoCo 3.10 contact witness uniquely, compares assembled constraint
rows and their Delassus block, then checks force/acceleration trajectories and
checkpoint replay. GPU execution is intentionally opt-in.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.bundled_plugins import lower_bundled_plugins
from mujoco_metal.plugin_sdf import MetalPluginSDFQuery
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.stepping import validate_stepping_profile

_PLUGIN_POSES = {
    "bolt": (-0.1, -0.1, -0.3),
    "bowl": (0.8, -0.6, 0.0),
    "gear": (0.9, 0.0, 0.0),
    "nut": (0.1, -0.4, -0.3),
    "torus": (0.3, -0.2, 0.0),
}
_BOX_VERTS = (
    "-0.05 -0.05 -0.05  0.05 -0.05 -0.05  0.05 0.05 -0.05 -0.05 0.05 -0.05 "
    "-0.05 -0.05 0.05  0.05 -0.05 0.05  0.05 0.05 0.05 -0.05 0.05 0.05")
_BOX_FACES = (
    "0 1 2 0 2 3 4 6 5 4 7 6 0 4 5 0 5 1 1 5 6 1 6 2 "
    "2 6 7 2 7 3 3 7 4 3 4 0")


def _plugin_sphere_model(name):
  x, y, z = _PLUGIN_POSES[name]
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="100" tolerance="1e-8" sdf_initpoints="8"/>
      <extension><plugin plugin="mujoco.sdf.{name}">
        <instance name="shape"/>
      </plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="probe" pos="{x} {y} {z}">
          <freejoint/><geom name="probe_sphere" type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <contact><pair geom1="plugin_sdf" geom2="probe_sphere" condim="1"/></contact>
    </mujoco>
  """)


def _plugin_mesh_sdf_model():
  # Upstream mjc_MeshSDF can emit up to mjMAXCONPAIR candidates regardless of
  # the seed setting. Keep its source-safe full 50-slot contract in this case.
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="100" tolerance="1e-8" sdf_initpoints="50"/>
      <extension><plugin plugin="mujoco.sdf.torus">
        <instance name="shape"/>
      </plugin></extension>
      <asset>
        <mesh name="shape_mesh"><plugin instance="shape"/></mesh>
        <mesh name="probe_mesh" vertex="{_BOX_VERTS}" face="{_BOX_FACES}"/>
      </asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="probe" pos="0.45 0 0">
          <freejoint/><geom name="probe_mesh_geom" type="mesh" mesh="probe_mesh"/>
        </body>
      </worldbody>
      <contact><pair geom1="probe_mesh_geom" geom2="plugin_sdf" condim="1"/></contact>
    </mujoco>
  """)


def _cpu_contacts(model, qpos, qvel):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  contacts = []
  J = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
  for i in range(data.ncon):
    contact = data.contact[i]
    count = 1 if int(contact.dim) == 1 else (
        2 * (int(contact.dim) - 1) if int(model.opt.cone) == 0 else int(contact.dim))
    base = int(contact.efc_address)
    contacts.append({
        "dist": float(contact.dist),
        "normal": np.asarray(contact.frame[:3], dtype=np.float64).copy(),
        "pos": np.asarray(contact.pos, dtype=np.float64).copy(),
        "rows": np.arange(base, base + count, dtype=np.int64),
    })
  return data, J, contacts


def _unique_native_slots(cpu_contacts, native_mask, native_pos):
  slots = np.flatnonzero(native_mask > 0.5)
  assert len(slots) == len(cpu_contacts), (len(slots), len(cpu_contacts))
  pairs = sorted(
      (float(np.linalg.norm(c["pos"] - native_pos[slot])), ci, int(slot))
      for ci, c in enumerate(cpu_contacts) for slot in slots)
  matched_cpu, matched_slots = set(), set()
  assignment = {}
  for distance, ci, slot in pairs:
    if ci in matched_cpu or slot in matched_slots:
      continue
    matched_cpu.add(ci)
    matched_slots.add(slot)
    assignment[ci] = (slot, distance)
  assert len(assignment) == len(cpu_contacts)
  assert max(distance for _, distance in assignment.values()) < 1e-3, assignment
  return assignment


def _assert_contact_system_matches_cpu(model, sim, cpu_data, cpu_J, cpu_contacts):
  desc = sim._coupled_constraints.descriptor
  system = sim.assembled_system(recompute=True)
  assert int(system["status"][0]) == 0
  mask = system["contact_mask"][0].cpu().numpy()
  positions = system["contact_position"][0].cpu().numpy()
  distances = system["contact_distance"][0].cpu().numpy()
  normals = system["contact_normal"][0].cpu().numpy()
  assignment = _unique_native_slots(cpu_contacts, mask, positions)
  native_J = system["J"][0].cpu().numpy()
  native_R = system["R"][0].cpu().numpy()
  native_ar = system["ar"][0].cpu().numpy()
  native_rhs = system["rhs"][0].cpu().numpy()
  packed = np.asarray(desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  native_rows = []
  cpu_rows = []
  for ci, contact in enumerate(cpu_contacts):
    slot, position_error = assignment[ci]
    assert position_error < 1e-3
    np.testing.assert_allclose(distances[slot], contact["dist"], rtol=2e-5, atol=5e-4)
    np.testing.assert_allclose(normals[slot], contact["normal"], rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(positions[slot], contact["pos"], rtol=1e-4, atol=1e-3)
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    gpu_rows = np.arange(gpu_base, gpu_base + len(contact["rows"]))
    source_rows = contact["rows"]
    np.testing.assert_allclose(native_J[gpu_rows], cpu_J[source_rows], rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(native_R[gpu_rows], cpu_data.efc_R[source_rows], rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(native_ar[gpu_rows], cpu_data.efc_aref[source_rows], rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(-native_rhs[gpu_rows], cpu_data.efc_b[source_rows], rtol=2e-4, atol=2e-3)
    native_rows.extend(gpu_rows.tolist())
    cpu_rows.extend(source_rows.tolist())

  native_rows = np.asarray(native_rows, dtype=np.int64)
  cpu_rows = np.asarray(cpu_rows, dtype=np.int64)
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, cpu_data, mass)
  cpu_W = cpu_J[cpu_rows] @ np.linalg.solve(mass, cpu_J[cpu_rows].T)
  np.testing.assert_allclose(
      system["W"][0].cpu().numpy()[np.ix_(native_rows, native_rows)],
      cpu_W, rtol=2e-5, atol=3e-6)
  np.testing.assert_allclose(
      system["qacc"][0].cpu().numpy(), cpu_data.qacc, rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(
      system["qfrc_constraint"][0].cpu().numpy(), cpu_data.qfrc_constraint,
      rtol=1e-3, atol=1e-3)
  return system


@pytest.mark.parametrize("name", tuple(_PLUGIN_POSES))
def test_bundled_rigid_plugin_sdf_source_contact_oracle_cpu(name):
  model = _plugin_sphere_model(name)
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert any("bundled bolt/bowl/gear/nut/torus SDF" in item
             for item in profile.supported)
  desc = lower_coupled_constraints(model)
  assert desc.ncontacts_max == 8 and desc.pair_max_contacts.tolist() == [8]
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qvel = np.zeros(model.nv, dtype=np.float64)
  cpu, J, contacts = _cpu_contacts(model, qpos, qvel)
  assert len(contacts) == 8 and all(c["dist"] < 0 for c in contacts)
  assert np.min([np.linalg.norm(a["pos"] - b["pos"])
                 for i, a in enumerate(contacts) for b in contacts[i + 1:]]) > 1e-5
  # Source rows are individually addressed, finite and unique before the
  # same fixture is admitted to the native gate.
  assert len(set(int(r) for c in contacts for r in c["rows"])) == 8
  assert np.all(np.isfinite(J))


def test_bundled_plugin_mesh_sdf_source_contact_oracle_cpu():
  model = _plugin_mesh_sdf_model()
  validate_stepping_profile(model, profile="integrated_euler_v1")
  desc = lower_coupled_constraints(model)
  assert desc.pair_max_contacts.tolist() == [50]
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qvel = np.zeros(model.nv, dtype=np.float64)
  cpu, J, contacts = _cpu_contacts(model, qpos, qvel)
  assert len(contacts) == 2 and min(c["dist"] for c in contacts) < -0.01
  assert np.all(np.isfinite(J))
  assert int(desc.mesh_hull_info.reshape(-1, 9)[0, 0]) == -2


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in bundled SDF workspace lifecycle check")
def test_bundled_sdf_query_reuse_clears_invalid_slots_and_replays_outputs():
  import torch

  model = _plugin_sphere_model("torus")
  descriptor = lower_bundled_plugins(model)
  query = MetalPluginSDFQuery(descriptor, batch_size=2, candidate_capacity=3)
  points = torch.tensor(
      [[[.61, .07, .13], [.5, 0., 0.], [.28, 0., 0.]],
       [[.61, .07, .13], [.5, 0., 0.], [0., 0., .11]]],
      dtype=torch.float32, device="mps")
  valid_ids = torch.zeros((2, 3), dtype=torch.int32, device="mps")
  first = query.run_device(points, valid_ids)
  first_values = {key: value.clone() for key, value in first.items()}
  assert first["status"].cpu().numpy().tolist() == [[0, 0, 0], [0, 0, 1]]
  assert torch.isfinite(first["distance"]).all()

  # Query-owned output storage is reused. Invalid and absent plugin IDs must
  # overwrite earlier nonzero results, while an invalid gradient is isolated
  # to its candidate and reported through the status buffer.
  invalid_ids = torch.tensor(
      [[-1, -1, -1], [9, 9, 9]], dtype=torch.int32, device="mps")
  second = query.run_device(points, invalid_ids)
  assert second["status"].cpu().numpy().tolist() == [[0, 0, 0], [1, 1, 1]]
  assert torch.equal(second["distance"][0], torch.zeros((3,), device="mps"))
  assert torch.equal(second["gradient"][0], torch.zeros((3, 3), device="mps"))
  assert torch.equal(second["distance"][1], torch.zeros((3,), device="mps"))
  assert torch.equal(second["gradient"][1], torch.zeros((3, 3), device="mps"))

  replay = query.run_device(points, valid_ids)
  for key, expected in first_values.items():
    assert torch.equal(replay[key], expected), key

  # Shape rejection is pre-dispatch and preserves the last valid workspace.
  with pytest.raises(ValueError, match="local_points"):
    query.run_device(points[:, :2], valid_ids)
  for key, expected in first_values.items():
    assert torch.equal(query.run_device(points, valid_ids)[key], expected), key


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in exact bundled SDF rigid force oracle")
@pytest.mark.parametrize("name", tuple(_PLUGIN_POSES))
def test_bundled_rigid_plugin_sdf_force_trajectory_checkpoint_matches_cpu(name):
  model = _plugin_sphere_model(name)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = sim.state._qpos[0].cpu().numpy().astype(np.float64, copy=True)
  qvel = sim.state._qvel[0].cpu().numpy().astype(np.float64, copy=True)
  cpu_data, cpu_J, contacts = _cpu_contacts(model, qpos, qvel)
  assert len(contacts) == 8
  _assert_contact_system_matches_cpu(model, sim, cpu_data, cpu_J, contacts)

  checkpoint = sim.snapshot()
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  first = []
  for _ in range(8):
    cpu_data.qfrc_applied[:] = qfrc[0]
    mujoco.mj_step(model, cpu_data)
    status = sim.step(1, qfrc_applied=qfrc)
    assert status.cpu().numpy().tolist() == [0]
    np.testing.assert_allclose(sim.state._qpos[0].cpu().numpy(), cpu_data.qpos,
                               rtol=1e-3, atol=2e-5)
    np.testing.assert_allclose(sim.state._qvel[0].cpu().numpy(), cpu_data.qvel,
                               rtol=1e-3, atol=2e-4)
    np.testing.assert_allclose(sim.state._qacc[0].cpu().numpy(), cpu_data.qacc,
                               rtol=1e-3, atol=1e-3)
    first.append((sim.state._qpos[0].clone(), sim.state._qvel[0].clone()))
  sim.restore(checkpoint)
  cpu_data, _, _ = _cpu_contacts(model, qpos, qvel)
  for expected_qpos, expected_qvel in first:
    cpu_data.qfrc_applied[:] = qfrc[0]
    mujoco.mj_step(model, cpu_data)
    assert sim.step(1, qfrc_applied=qfrc).cpu().numpy().tolist() == [0]
    assert sim.state._qpos[0].equal(expected_qpos)
    assert sim.state._qvel[0].equal(expected_qvel)

  sim.reset(qpos=qpos.astype(np.float32)[None],
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  reset_cpu, reset_J, reset_contacts = _cpu_contacts(model, qpos, qvel)
  _assert_contact_system_matches_cpu(model, sim, reset_cpu, reset_J,
                                     reset_contacts)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in plugin/mesh-SDF rigid force oracle")
def test_bundled_plugin_mesh_sdf_rigid_rows_force_and_replay_match_cpu():
  model = _plugin_mesh_sdf_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = sim.state._qpos[0].cpu().numpy().astype(np.float64, copy=True)
  qvel = sim.state._qvel[0].cpu().numpy().astype(np.float64, copy=True)
  cpu_data, cpu_J, contacts = _cpu_contacts(model, qpos, qvel)
  assert len(contacts) == 2
  _assert_contact_system_matches_cpu(model, sim, cpu_data, cpu_J, contacts)

  checkpoint = sim.snapshot()
  force = np.zeros((1, model.nv), dtype=np.float32)
  first = []
  for _ in range(8):
    cpu_data.qfrc_applied[:] = force[0]
    mujoco.mj_step(model, cpu_data)
    assert sim.step(1, qfrc_applied=force).cpu().numpy().tolist() == [0]
    np.testing.assert_allclose(sim.state._qpos[0].cpu().numpy(), cpu_data.qpos,
                               rtol=1e-3, atol=2e-5)
    np.testing.assert_allclose(sim.state._qvel[0].cpu().numpy(), cpu_data.qvel,
                               rtol=1e-3, atol=2e-4)
    first.append((sim.state._qpos[0].clone(), sim.state._qvel[0].clone()))
  sim.restore(checkpoint)
  cpu_data, _, _ = _cpu_contacts(model, qpos, qvel)
  for expected_qpos, expected_qvel in first:
    mujoco.mj_step(model, cpu_data)
    assert sim.step(1, qfrc_applied=force).cpu().numpy().tolist() == [0]
    assert sim.state._qpos[0].equal(expected_qpos)
    assert sim.state._qvel[0].equal(expected_qvel)

  sim.reset(qpos=qpos.astype(np.float32)[None],
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  reset_cpu, reset_J, reset_contacts = _cpu_contacts(model, qpos, qvel)
  _assert_contact_system_matches_cpu(model, sim, reset_cpu, reset_J,
                                     reset_contacts)
