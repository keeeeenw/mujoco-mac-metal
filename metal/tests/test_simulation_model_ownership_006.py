# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Model ownership at preparation and explicit rebuild boundaries."""
import copy
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation


_XML = '''<mujoco><option timestep=".003" gravity="0 0 -9.81">
  <flag contact="disable"/></option><worldbody><body pos="0 0 .5">
  <joint type="slide" axis="0 0 1" damping=".7"/>
  <geom type="sphere" size=".1" mass="2"/>
  </body></worldbody></mujoco>'''


def test_constructor_owns_model_before_device_preparation(monkeypatch):
  import mujoco_metal.simulation as module
  source = mujoco.MjModel.from_xml_string(_XML)
  captured = []
  class Prepared(Exception):
    pass
  def stop_before_device(model, *args, **kwargs):
    captured.append(model)
    raise Prepared
  monkeypatch.setattr(module, "DeviceState", stop_before_device)
  with pytest.raises(Prepared):
    MetalSimulation(source, 2, profile="contact_free_forces_euler_v1")
  owned = captured[0]
  assert owned is not source
  np.testing.assert_array_equal(owned.body_mass, source.body_mass)
  source.opt.timestep = .5
  source.body_mass[1] = 17
  assert owned.opt.timestep == .003
  assert owned.body_mass[1] == 2
  owned.opt.gravity[:] = 0
  np.testing.assert_array_equal(source.opt.gravity, [0, 0, -9.81])


def test_model_inspection_is_an_owned_copy():
  sim = object.__new__(MetalSimulation)
  sim._mjmodel = mujoco.MjModel.from_xml_string(_XML)
  first, second = sim.model, sim.model
  assert first is not sim._mjmodel and second is not first
  first.opt.timestep = .7
  first.dof_damping[:] = 11
  assert sim._mjmodel.opt.timestep == second.opt.timestep == .003
  np.testing.assert_array_equal(sim._mjmodel.dof_damping, [.7])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_model_mutation_isolation_replay_and_explicit_rebuild():
  from mujoco_metal.lifecycle import ModelLifecycle
  source = mujoco.MjModel.from_xml_string(_XML)
  oracle = copy.copy(source)
  qpos = np.array([[.125], [-.25]], np.float32)
  qvel = np.array([[.2], [-.3]], np.float32)
  force = np.array([[.3], [-.4]], np.float32)
  sim = MetalSimulation(source, 2, qpos=qpos, qvel=qvel,
                        profile="contact_free_forces_euler_v1")
  refs = [mujoco.MjData(oracle) for _ in range(2)]
  for row, data in enumerate(refs):
    data.qpos[:] = qpos[row]
    data.qvel[:] = qvel[row]
    data.qfrc_applied[:] = force[row]
  # Neither the constructor's caller model nor an inspection copy can change
  # a live step's timestep/forces independently of its prepared device mass.
  for model in (source, sim.model):
    model.opt.timestep = .9
    model.opt.gravity[:] = [31, -7, 21]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.body_mass[1] = 17
    model.dof_damping[:] = 23
  for _ in range(8):
    sim.step(qfrc_applied=force)
    for row, data in enumerate(refs):
      mujoco.mj_step(oracle, data)
      for field in ("qpos", "qvel", "qacc", "time"):
        actual = getattr(sim.state, "_"+field)[row].cpu().numpy()
        np.testing.assert_allclose(actual, getattr(data, field), rtol=2e-6, atol=2e-6)
  checkpoint = sim.snapshot()
  sim.step(4)
  completed = sim.snapshot()["device"]
  sim.restore(checkpoint)
  sim.step(4)
  replay = sim.snapshot()["device"]
  for field in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(getattr(replay, field), getattr(completed, field))
  life = ModelLifecycle(oracle)
  assert life.recompute_body_masses([1], [3.])
  sim.apply_lifecycle(life)
  assert sim.model.body_mass[1] == 3
  rebuilt = sim.model
  before = sim.state.snapshot()
  life._model.opt.timestep = .75
  life._model.body_mass[1] = 29
  sim.step(qfrc_applied=force)
  for row in range(2):
    reference = mujoco.MjData(rebuilt)
    reference.qpos[:] = before.qpos[row]
    reference.qvel[:] = before.qvel[row]
    reference.time = float(before.time[row])
    reference.qfrc_applied[:] = force[row]
    mujoco.mj_step(rebuilt, reference)
    for field in ("qpos", "qvel", "qacc", "time"):
      np.testing.assert_allclose(getattr(sim.state, "_"+field)[row].cpu().numpy(),
                                 getattr(reference, field), rtol=2e-6, atol=2e-6)
  np.testing.assert_array_equal(sim.state._status.cpu().numpy(), [0, 0])
