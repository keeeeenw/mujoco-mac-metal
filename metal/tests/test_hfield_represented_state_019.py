"""Separate source oracle for the public float32 native initial state.

The CPU default-pose fixture retains its full compiled qpos0. Native parity
selectors in ``test_mesh_hfield_contact_012.py`` supply the identical explicit
public pose to both engines. These tests retain the distinct double-default
witness and independently exercise the complete represented-state oracle.
"""
import os

import mujoco
import numpy as np
import pytest

from test_mesh_hfield_contact_012 import (
    _LIMITS, _model, _native_contacts, _source_contacts,
)
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.simulation import MetalSimulation


def _source_at_qpos(model, qpos):
  """Run actual pinned mj_forward at the explicit represented qpos."""
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(qpos, dtype=np.float64)
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  contacts = [data.contact[i] for i in range(data.ncon)
              if set(map(int, data.contact[i].geom)) == {terrain, mesh}]
  J = np.asarray(data.efc_J).reshape(int(data.nefc), model.nv).copy()
  return data, contacts, J, np.asarray(data.efc_R).copy(), np.asarray(
      data.efc_aref).copy()


def test_default_and_explicit_native_qpos_share_float32_source_oracle_cpu():
  """Pin the default qpos0 cast and show why original-double stays separate."""
  model = _model(condim=1, cone="elliptic")
  double_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  public_qpos = double_qpos.astype(np.float32)
  represented_qpos = public_qpos.astype(np.float64)
  assert double_qpos[2] == 0.03
  assert represented_qpos[2] == float(np.float32(0.03))
  assert represented_qpos[2] != double_qpos[2]

  original, original_contacts = _source_contacts(model)
  represented, represented_contacts, represented_J, represented_R, represented_ar = (
      _source_at_qpos(model, represented_qpos))
  assert len(original_contacts) == len(represented_contacts) == 30
  assert represented_J.shape == (30, model.nv)
  assert represented_R.shape == represented_ar.shape == (30,)
  assert np.all(np.isfinite(represented_J))
  assert np.all(np.isfinite(represented_R))
  assert np.all(np.isfinite(represented_ar))

  # This exact contact demonstrates the discrete support/EPA branch change
  # from one float32 input ulp. It is diagnostic evidence, not a relaxed bound.
  assert float(original_contacts[8].dist) == pytest.approx(
      -0.014069049810730497, abs=2e-14)
  assert float(represented_contacts[8].dist) == pytest.approx(
      -0.019999997317791299, abs=2e-14)
  assert abs(float(original_contacts[8].dist)
             - float(represented_contacts[8].dist)) > 5e-3

  explicit, explicit_contacts = _source_contacts(model, qpos=public_qpos)
  np.testing.assert_array_equal(explicit.qpos, represented_qpos)
  np.testing.assert_array_equal(explicit.efc_J, represented.efc_J)
  np.testing.assert_array_equal(explicit.efc_R, represented_R)
  np.testing.assert_array_equal(explicit.efc_aref, represented_ar)
  assert len(explicit_contacts) == len(represented_contacts) == 30
  for actual, expected in zip(explicit_contacts, represented_contacts):
    assert float(actual.dist) == float(expected.dist)
    np.testing.assert_array_equal(actual.pos, expected.pos)
    np.testing.assert_array_equal(actual.frame, expected.frame)

  # A full pinned step and checkpoint copy/replay at the represented state are
  # deterministic on CPU. This is the same state handed to native reset below.
  mujoco.mj_step(model, represented)
  checkpoint = mujoco.MjData(model)
  mujoco.mj_copyData(checkpoint, model, represented)
  replay = mujoco.MjData(model)
  mujoco.mj_copyData(replay, model, checkpoint)
  np.testing.assert_array_equal(replay.qpos, represented.qpos)
  np.testing.assert_array_equal(replay.qvel, represented.qvel)
  np.testing.assert_array_equal(replay.qacc, represented.qacc)
  assert replay.time == represented.time


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in exact represented-state HField parity")
def test_native_hfield_all_contacts_rows_force_and_replay_at_represented_qpos_gpu():
  """Compare full public native HField outputs to pinned same-input source."""
  model = _model(condim=1, cone="elliptic")
  public_qpos = np.asarray(model.qpos0, dtype=np.float32)
  qpos64 = public_qpos.astype(np.float64)
  cpu, source_contacts, source_J, source_R, source_ar = _source_at_qpos(
      model, qpos64)
  assert len(source_contacts) == 30

  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1",
                        limits=_LIMITS)
  # Constructor defaults and explicit reset must converge on the same public
  # float32 qpos representation of compiled model.qpos0.
  np.testing.assert_array_equal(
      sim.state._qpos.detach().cpu().numpy(), public_qpos[None])
  sim.reset(qpos=public_qpos[None],
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  np.testing.assert_array_equal(
      sim.state._qpos.detach().cpu().numpy(), public_qpos[None])
  checkpoint = sim.snapshot()
  assembly = sim.assembled_system(recompute=True)
  native_contacts = _native_contacts(assembly)
  assert len(native_contacts) == len(source_contacts) == 30

  # Keep the strict tolerances from the primary parity test. The
  # source data input differs: this oracle receives the public float32 qpos.
  for index, (actual, source) in enumerate(zip(native_contacts, source_contacts)):
    distance, position, normal = actual
    assert abs(distance - float(source.dist)) <= 6e-6, index
    np.testing.assert_allclose(position, source.pos, rtol=0, atol=6e-6,
                               err_msg=f"represented contact {index} position")
    np.testing.assert_allclose(normal, source.frame[:3], rtol=0, atol=6e-5,
                               err_msg=f"represented contact {index} normal")

  desc = lower_coupled_constraints(model, limits=_LIMITS)
  row_start = int(desc.nr_joint)
  row_stop = row_start + len(source_contacts)
  native_J = assembly["J"].detach().cpu().numpy()[0, row_start:row_stop]
  np.testing.assert_allclose(native_J, source_J, rtol=0, atol=6e-6,
                             err_msg="represented canonical contact J")
  native_R = assembly["R"].detach().cpu().numpy()[0, row_start:row_stop]
  native_ar = assembly["ar"].detach().cpu().numpy()[0, row_start:row_stop]
  np.testing.assert_allclose(native_R, source_R, rtol=1e-4, atol=6e-6,
                             err_msg="represented contact R")
  np.testing.assert_allclose(native_ar, source_ar, rtol=2e-4, atol=6e-5,
                             err_msg="represented contact aref")
  np.testing.assert_allclose(
      assembly["qfrc_constraint"].detach().cpu().numpy()[0],
      cpu.qfrc_constraint, rtol=2e-3, atol=3e-2,
      err_msg="represented initial constraint force")

  # Compare the actual public step and then replay the exact MPS checkpoint.
  mujoco.mj_step(model, cpu)
  sim.step(1)
  stepped = {name: getattr(sim.state, "_" + name).detach().cpu().numpy().copy()
             for name in ("qpos", "qvel", "qacc", "time")}
  np.testing.assert_allclose(stepped["qpos"][0], cpu.qpos, rtol=2e-5,
                             atol=3e-5)
  np.testing.assert_allclose(stepped["qvel"][0], cpu.qvel, rtol=2e-4,
                             atol=5e-4)
  np.testing.assert_allclose(stepped["qacc"][0], cpu.qacc, rtol=2e-3,
                             atol=3e-2)
  assert stepped["time"][0] == pytest.approx(float(cpu.time), abs=1e-8)
  sim.restore(checkpoint)
  sim.step(1)
  for name, expected in stepped.items():
    np.testing.assert_array_equal(
        getattr(sim.state, "_" + name).detach().cpu().numpy(), expected,
        err_msg=f"represented checkpoint replay {name}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in represented-state healthy-neighbor recovery")
def test_native_hfield_failed_world_keeps_represented_healthy_trajectory_gpu():
  """Use exact public reset values in source healthy-neighbor isolation."""
  model = _model(condim=1, cone="elliptic")
  initial = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2,
                      axis=0)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1",
                        limits=_LIMITS)
  sim.reset(qpos=initial, qvel=np.zeros((2, model.nv), np.float32))
  checkpoint = sim.snapshot()
  mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  program = sim._coupled_constraints
  original_generate = program.generate_candidates

  def inject_invalid_first_world(poses, qvel, eq_active=None, **kwargs):
    modified = dict(poses)
    modified["geom_quat"] = poses["geom_quat"].clone()
    modified["geom_quat"][0, mesh, 0] = float("nan")
    return original_generate(modified, qvel, eq_active, **kwargs)

  program.generate_candidates = inject_invalid_first_world
  try:
    status = sim.step(1)
  finally:
    program.generate_candidates = original_generate
  assert status.detach().cpu().numpy().tolist() == [2, 0]

  expected = mujoco.MjData(model)
  expected.qpos[:] = initial[1].astype(np.float64)
  expected.qvel[:] = 0
  mujoco.mj_step(model, expected)
  np.testing.assert_array_equal(
      sim.state._qpos[0].detach().cpu().numpy(), initial[0])
  np.testing.assert_array_equal(
      sim.state._qvel[0].detach().cpu().numpy(), np.zeros(model.nv, np.float32))
  np.testing.assert_allclose(sim.state._qpos[1].detach().cpu().numpy(),
                             expected.qpos, rtol=2e-5, atol=3e-5)
  np.testing.assert_allclose(sim.state._qvel[1].detach().cpu().numpy(),
                             expected.qvel, rtol=2e-4, atol=5e-4)
  np.testing.assert_allclose(sim.state._qacc[1].detach().cpu().numpy(),
                             expected.qacc, rtol=2e-3, atol=3e-2)
  assert float(sim.state._time[1].detach().cpu()) == pytest.approx(
      float(expected.time), abs=1e-8)

  # Checkpoint restore followed by the ordinary public path must be healthy and
  # reproduce both represented source trajectories exactly.
  sim.restore(checkpoint)
  replay_status = sim.step(1)
  assert replay_status.detach().cpu().numpy().tolist() == [0, 0]
  for world in range(2):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = initial[world].astype(np.float64)
    cpu.qvel[:] = 0
    mujoco.mj_step(model, cpu)
    np.testing.assert_allclose(sim.state._qpos[world].cpu().numpy(), cpu.qpos,
                               rtol=2e-5, atol=3e-5)
    np.testing.assert_allclose(sim.state._qvel[world].cpu().numpy(), cpu.qvel,
                               rtol=2e-4, atol=5e-4)
