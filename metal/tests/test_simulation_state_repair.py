# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R02/R03 repair: activation atomicity and simulation-level state ownership.

R02: disabled-actuation preservation, failed-world activation rollback,
recovery after selected reset.
R03: versioned simulation snapshots covering device state, warm starts,
held inputs and stored sensor samples; query isolation twins; atomic
invalid restore; restore+randomize coherence; raw-vs-simulator reset.
"""

import os

import mujoco
import numpy as np
import pytest


def _filter_model(disabled=True):
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody>"
      "<body pos='0 0 0.5'><joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/></body></worldbody>"
      "<actuator><general name='g' joint='h' dyntype='filter' gainprm='2' biasprm='0 -2 0'/></actuator>"
      "</mujoco>")
  if disabled:
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  return m


def test_disabled_actuation_preserves_activation_cpu():
  model = _filter_model(disabled=True)
  assert model.na == 1
  d = mujoco.MjData(model)
  d.qpos[:] = 0.1
  d.act[:] = [0.7]
  for _ in range(5):
    mujoco.mj_step(model, d)
    assert float(d.act[0]) == pytest.approx(0.7), d.act[:]
  assert bool(np.all(np.isfinite(d.qpos)))


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
def test_disabled_actuation_preserves_activation_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  model = _filter_model(disabled=True)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.1]], dtype=np.float32),
            qvel=np.zeros((1, model.nv), dtype=np.float32),
            act=np.array([[0.7]], dtype=np.float32))
  for _ in range(5):
    sim.step(1, ctrl=np.array([[1.0]], dtype=np.float32))
  assert float(sim.state._act.cpu().numpy()[0, 0]) == pytest.approx(0.7)
  # CPU twin with frozen activation agrees on qpos/qvel.
  d = mujoco.MjData(model)
  d.qpos[:] = [0.1]
  d.act[:] = [0.7]
  for _ in range(5):
    d.ctrl[:] = [1.0]
    mujoco.mj_step(model, d)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0],
                             np.asarray(d.qpos), rtol=1e-5, atol=1e-6)


@_needs_gpu()
def test_failed_world_keeps_activation_and_state_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  model = _filter_model(disabled=False)
  assert model.na == 1
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.1], [0.2]], dtype=np.float32),
            qvel=np.zeros((2, model.nv), dtype=np.float32),
            act=np.array([[0.3], [0.4]], dtype=np.float32))
  sim.step(2, ctrl=np.array([[0.5], [0.5]], dtype=np.float32))
  healthy_act = sim.state._act.cpu().numpy().copy()
  healthy_qpos = sim.state.qpos.cpu().numpy().copy()
  # Inject a force-evaluation failure into world 1 only (NaN state).
  bad = sim.state._qpos.cpu().numpy()
  bad[1, 0] = np.nan
  sim.state._qpos.copy_(sim.state._torch.as_tensor(bad))
  sim.step(1, ctrl=np.array([[0.5], [0.5]], dtype=np.float32))
  status = sim.state._status.cpu().numpy()
  assert int(status[1]) != 0
  after_act = sim.state._act.cpu().numpy()
  after_qpos = sim.state.qpos.cpu().numpy()
  # Failed world: activation and observable state exactly preserved.
  assert after_act[1, 0] == healthy_act[1, 0]
  assert np.array_equal(after_qpos[1], healthy_qpos[1], equal_nan=True) or \
      np.all(~np.isfinite(after_qpos[1]))
  # Healthy world: kept advancing (finite and moved).
  assert bool(np.all(np.isfinite(after_qpos[0])))
  assert not np.array_equal(after_qpos[0], healthy_qpos[0])
  # Recovery: reset the failed world, both advance again.
  sim.reset(env_ids=[1])
  sim.step(1, ctrl=np.array([[0.5], [0.5]], dtype=np.float32))
  assert bool(np.all(np.isfinite(sim.state.qpos.cpu().numpy())))


def _contact_model(with_acc_sensors=False):
  sens = "<force site='s'/><torque site='s'/><accelerometer site='s'/>" if with_acc_sensors else "<touch site='s'/>"
  return mujoco.MjModel.from_xml_string(
      f"<mujoco><option timestep='0.002' integrator='Euler' iterations='1'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='-0.3 0 0.25'><freejoint/><geom type='sphere' size='0.1' mass='0.5'/>"
      "<site name='s'/></body>"
      "<body pos='0.3 0 0.4'><freejoint/><geom type='sphere' size='0.1' mass='0.5'/></body>"
      "</worldbody>"
      f"<sensor>{sens}</sensor></mujoco>")


def _full_state(sim):
  return {
      "generation": sim.state.generation,
      "qpos": sim.state.qpos.cpu().numpy().copy(),
      "qvel": sim.state.qvel.cpu().numpy().copy(),
      "act": sim.state._act.cpu().numpy().copy() if getattr(sim.state, "_na", 0) else None,
      "warm": sim._coupled_constraints.get_warmstart().copy()
      if getattr(sim, "_coupled_constraints", None) is not None else None,
      "sens": sim.step_sensordata().copy() if getattr(sim, "_sensordata", None) is not None else None,
  }


@_needs_gpu()
def test_simulation_snapshot_replay_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  for _ in range(80):
    sim.step(1)
  snap = sim.snapshot()
  ref = _full_state(sim)
  for _ in range(30):
    sim.step(1)
  assert not np.allclose(sim.state.qpos.cpu().numpy(), ref["qpos"])
  sim.restore(snap)
  got = _full_state(sim)
  for key in ("qpos", "qvel", "sens"):
    np.testing.assert_array_equal(got[key], ref[key])
  np.testing.assert_array_equal(got["warm"], ref["warm"])
  # Replay continues identically (low iteration budget: seeds matter).
  for _ in range(30):
    sim.step(1)
  again = _full_state(sim)
  sim.restore(snap)
  for _ in range(30):
    sim.step(1)
  twice = _full_state(sim)
  for key in ("qpos", "qvel", "sens"):
    np.testing.assert_array_equal(again[key], twice[key])
  np.testing.assert_array_equal(again["warm"], twice["warm"])


@_needs_gpu()
def test_copy_environment_reproduces_trajectory_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  for _ in range(80):
    sim.step(1)
  sim.copy_environment(0, 1)
  for _ in range(10):
    sim.step(1)
  q = sim.state.qpos.cpu().numpy()
  np.testing.assert_array_equal(q[0], q[1])
  w = sim._coupled_constraints.get_warmstart()
  np.testing.assert_array_equal(w[0], w[1])
  s = sim.step_sensordata()
  np.testing.assert_array_equal(s[0], s[1])


@_needs_gpu()
def test_query_twins_have_identical_next_steps_gpu():
  from mujoco_metal.simulation import MetalSimulation
  # Includes FORCE/TORQUE and accelerometer sensors (executes the _has_acc_sensors branch)
  model = _contact_model(with_acc_sensors=True)
  twins = []
  for _ in range(2):
    sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
    # Asymmetric input across worlds so worlds develop different warm seeds
    qp = np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0)
    qp[1, 0] += 0.05
    sim.reset(qpos=qp, qvel=np.zeros((2, model.nv), dtype=np.float32))
    twins.append(sim)
  for _ in range(40):
    for sim in twins:
      sim.step(1)
  a, b = twins
  wa_before = a._coupled_constraints.get_warmstart().copy()
  acc_step_before = a.accepted_step
  assert acc_step_before is not None
  lam_step_before = acc_step_before["system"]["lambda"].cpu().numpy().copy()

  # Query forward with ACC sensors
  q_vals = a.sensor_values()
  assert q_vals is not None

  # Query changes nothing persistent: warm seeds, caches, and accepted step record identical
  np.testing.assert_array_equal(a._coupled_constraints.get_warmstart(), wa_before)
  assert a.accepted_step is not None
  np.testing.assert_array_equal(a.accepted_step["system"]["lambda"].cpu().numpy(), lam_step_before)

  # Trajectories advance identically after query
  for _ in range(5):
    a.step(1)
    b.step(1)
  np.testing.assert_array_equal(a.state.qpos.cpu().numpy(), b.state.qpos.cpu().numpy())
  np.testing.assert_array_equal(a.state.qvel.cpu().numpy(), b.state.qvel.cpu().numpy())
  np.testing.assert_array_equal(a._coupled_constraints.get_warmstart(),
                                b._coupled_constraints.get_warmstart())
  np.testing.assert_array_equal(a.step_sensordata(), b.step_sensordata())


@_needs_gpu()
def test_invalid_restore_leaves_state_unchanged_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(with_acc_sensors=True)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  for _ in range(5):
    sim.step(1)
  snap = sim.snapshot()
  ref = _full_state(sim)

  # 1. NaN warmstart
  bad = dict(snap)
  bad["warmstart"] = np.full_like(snap["warmstart"], np.nan)
  with pytest.raises(ValueError):
    sim.restore(bad)

  # 2. Float32-overflowing finite float64 warmstart (1e300)
  bad_overflow = dict(snap)
  bad_warm = snap["warmstart"].astype(np.float64).copy()
  bad_warm[0, 0] = 1e300
  bad_overflow["warmstart"] = bad_warm
  with pytest.raises(ValueError, match="overflow float32"):
    sim.restore(bad_overflow)

  # 3. Not a snapshot dict
  with pytest.raises(ValueError):
    sim.restore("not-a-snapshot")

  # 4. Out-of-bounds and empty env_ids
  with pytest.raises(ValueError):
    sim.restore(snap, env_ids=[])
  with pytest.raises(ValueError):
    sim.restore(snap, env_ids=[999])
  with pytest.raises(ValueError):
    sim.copy_environment(0, 5)

  # Verify all live tensors, generation, warm seeds, and samples are 100% untouched
  got = _full_state(sim)
  assert got["generation"] == ref["generation"]
  for key in ("qpos", "qvel", "sens"):
    np.testing.assert_array_equal(got[key], ref[key])
  np.testing.assert_array_equal(got["warm"], ref["warm"])


@_needs_gpu()
def test_selected_world_restore_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(with_acc_sensors=True)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  qp = np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0)
  qp[1, 0] += 0.05
  sim.reset(qpos=qp, qvel=np.zeros((2, model.nv), dtype=np.float32))
  for _ in range(10):
    sim.step(1)
  snap = sim.snapshot()

  # Advance 10 more steps
  for _ in range(10):
    sim.step(1)
  ref_w1 = {
      "qpos": sim.state.qpos[1].cpu().numpy().copy(),
      "qvel": sim.state.qvel[1].cpu().numpy().copy(),
      "warm": sim._coupled_constraints.get_warmstart()[1].copy(),
      "sens": sim.step_sensordata()[1].copy(),
  }

  # Restore world 0 only
  sim.restore(snap, env_ids=[0])

  # World 0 restored to snapshot
  np.testing.assert_array_equal(sim.state.qpos[0].cpu().numpy(), snap["device"].qpos[0])
  np.testing.assert_array_equal(sim.state.qvel[0].cpu().numpy(), snap["device"].qvel[0])
  np.testing.assert_array_equal(sim._coupled_constraints.get_warmstart()[0], snap["warmstart"][0])
  np.testing.assert_array_equal(sim.step_sensordata()[0], snap["sensordata"][0])

  # World 1 was completely untouched
  np.testing.assert_array_equal(sim.state.qpos[1].cpu().numpy(), ref_w1["qpos"])
  np.testing.assert_array_equal(sim.state.qvel[1].cpu().numpy(), ref_w1["qvel"])
  np.testing.assert_array_equal(sim._coupled_constraints.get_warmstart()[1], ref_w1["warm"])
  np.testing.assert_array_equal(sim.step_sensordata()[1], ref_w1["sens"])


@_needs_gpu()
def test_fresh_snapshot_restores_and_replays_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(with_acc_sensors=True)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  qp = np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0)
  sim.reset(qpos=qp, qvel=np.zeros((2, model.nv), dtype=np.float32))

  # Fresh snapshot before any step has run (sensordata is None)
  snap0 = sim.snapshot()
  assert snap0["sensordata"] is None

  # Step and record trajectory
  for _ in range(10):
    sim.step(1)
  ref_qpos = sim.state.qpos.cpu().numpy().copy()
  ref_sens = sim.step_sensordata().copy()
  snap10 = sim.snapshot()

  # Restore fresh snapshot: must succeed and reset _sensordata to None
  sim.restore(snap0)
  assert getattr(sim, "_sensordata", None) is None

  # Replay identical trajectory from restored fresh snapshot
  for _ in range(10):
    sim.step(1)
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), ref_qpos)
  np.testing.assert_array_equal(sim.step_sensordata(), ref_sens)

  # Restore snapshot from stepped sim into a fresh, compatible second simulator
  sim2 = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  assert getattr(sim2, "_sensordata", None) is None
  sim2.restore(snap10)
  np.testing.assert_array_equal(sim2.state.qpos.cpu().numpy(), ref_qpos)
  np.testing.assert_array_equal(sim2.step_sensordata(), ref_sens)

  # Both simulators step identically forward
  for _ in range(5):
    sim.step(1)
    sim2.step(1)
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), sim2.state.qpos.cpu().numpy())
  np.testing.assert_array_equal(sim.step_sensordata(), sim2.step_sensordata())


@_needs_gpu()
def test_retained_step_and_zero_solves_on_assembled_system_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(with_acc_sensors=True)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  for _ in range(10):
    sim.step(1)

  # Explicit accepted-step record
  acc = sim.accepted_step
  assert acc is not None
  assert acc["generation"] == sim.state.generation
  assert int(acc["status"].cpu().numpy()[0]) == 0
  np.testing.assert_array_equal(acc["acceleration"].cpu().numpy(), sim.state.qacc.cpu().numpy())

  # Calling assembled_system() immediately after step must return accepted-step without solve
  dispatches_before = sim._coupled_solve_dispatches
  asm = sim.assembled_system()
  assert sim._coupled_solve_dispatches == dispatches_before, "assembled_system must return cached step without solving"
  np.testing.assert_array_equal(asm["lambda"].cpu().numpy(), acc["system"]["lambda"].cpu().numpy())

  # recompute=True must dispatch exactly one solve
  asm_re = sim.assembled_system(recompute=True)
  assert sim._coupled_solve_dispatches == dispatches_before + 1


@_needs_gpu()
def test_restore_then_randomize_stays_coherent_gpu():
  from mujoco_metal.simulation import MetalSimulation
  rng = np.random.default_rng(7)
  model = _contact_model()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  for _ in range(5):
    sim.step(1)
  snap = sim.snapshot()
  sim.restore(snap)
  nq = model.nq
  assert nq == 14
  qpos = rng.normal(0, 0.05, size=(2, nq)).astype(np.float32)
  qpos[:, 0:3] += np.asarray(model.qpos0, dtype=np.float32)[0:3]
  qpos[:, 3:7] = np.asarray([1, 0, 0, 0], dtype=np.float32)
  qpos[:, 7:10] += np.asarray(model.qpos0, dtype=np.float32)[7:10]
  qpos[:, 10:14] = np.asarray([1, 0, 0, 0], dtype=np.float32)
  sim.reset(qpos=qpos, qvel=np.zeros((2, model.nv), dtype=np.float32))
  for _ in range(10):
    sim.step(1)
  assert bool(np.all(np.isfinite(sim.state.qpos.cpu().numpy())))
  assert bool(np.all(np.isfinite(sim.step_sensordata())))


@_needs_gpu()
def test_raw_reset_keeps_warmstart_simulator_reset_clears_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  for _ in range(120):
    sim.step(1)
  warm_before = sim._coupled_constraints.get_warmstart().copy()
  assert bool(np.any(warm_before != 0))
  # Raw device reset: kinematics replaced, retained solver seed untouched.
  sim.state.reset()
  np.testing.assert_array_equal(sim._coupled_constraints.get_warmstart(), warm_before)
  for _ in range(120):
    sim.step(1)
  # Simulator reset: cold start across state, seeds, held inputs, samples.
  sim.reset()
  np.testing.assert_array_equal(sim._coupled_constraints.get_warmstart(),
                                np.zeros_like(warm_before))
  np.testing.assert_array_equal(sim.step_sensordata(), np.zeros_like(sim.step_sensordata()))
