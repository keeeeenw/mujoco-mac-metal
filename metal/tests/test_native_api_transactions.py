# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned mjtState ownership, transaction, and cache-coherence regressions."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal.native_api import StateSpec, mj_getState, mj_setState

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
]


def _model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="PGS" iterations="100"/>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>""")


def _host(t):
  return t.detach().cpu().numpy().copy()


def test_applied_force_selector_is_owned_and_drives_step():
  model = _model()
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  mj_setState(sim, {"qfrc_applied": np.array([[2]], np.float32)})
  assert "qfrc_applied" in mj_getState(sim, StateSpec.QFRC_APPLIED)
  sim.step()
  np.testing.assert_allclose(_host(sim.state.qacc), [[2]], atol=1e-6)
  # Omitted arguments retain installed native state; explicit zero clears it.
  sim.step()
  np.testing.assert_allclose(_host(sim.state.qacc), [[2]], atol=1e-6)
  sim.step(qfrc_applied=np.zeros((1, model.nv), np.float32))
  np.testing.assert_allclose(_host(sim.state.qacc), [[0]], atol=1e-6)


def test_setstate_is_atomic_and_invalidates_assembled_cache():
  model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".002" solver="PGS" iterations="100"/>
    <worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .09">
      <joint type="slide" axis="0 0 1"/><geom type="sphere" size=".1" mass="1" condim="1"/>
    </body></worldbody></mujoco>""")
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  sim.step()
  before = _host(sim.state.qpos)
  generation = sim.state.generation
  bad = np.full((1, model.nv), np.nan, np.float32)
  with pytest.raises(ValueError):
    mj_setState(sim, {"qpos": np.array([[.7]], np.float32), "warmstart": bad})
  np.testing.assert_array_equal(_host(sim.state.qpos), before)
  assert sim.state.generation == generation

  mj_setState(sim, {"qpos": np.array([[1.0]], np.float32)})
  assert sim.state.generation == generation + 1
  assert not sim._assembled_system_valid
  np.testing.assert_allclose(_host(sim.assembled_system()["qfrc_constraint"]), 0, atol=1e-5)


def test_selected_full_state_groups_copy_reset_and_restore():
  model = _model()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  values = {
      "time": np.array([.25, .5], np.float32),
      "qpos": np.array([[.1], [.2]], np.float32),
      "qvel": np.array([[.3], [.4]], np.float32),
      "qacc": np.array([[.5], [.6]], np.float32),
      "qacc_warmstart": np.array([[.7], [.8]], np.float32),
      "qfrc_applied": np.array([[1.], [2.]], np.float32),
      "userdata": np.empty((2, 0), np.float32),
      "plugin_state": np.empty((2, 0), np.float32),
  }
  mj_setState(sim, values)
  state = mj_getState(sim, StateSpec.INTEGRATION)
  assert set(state) == {
      "time", "qpos", "qvel", "act", "history", "warmstart", "ctrl",
      "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos",
      "mocap_quat", "userdata", "plugin_state",
  }
  assert state["warmstart"].shape == (2, model.nv)
  np.testing.assert_allclose(_host(state["warmstart"]), values["qacc_warmstart"])
  np.testing.assert_allclose(_host(state["qfrc_applied"]), values["qfrc_applied"])
  roundtrip = mj_getState(sim, StateSpec.ALL)
  mj_setState(sim, roundtrip)
  snap = sim.snapshot()

  mj_setState(sim, {"qpos": np.array([[9.]], np.float32)}, env_ids=[1])
  np.testing.assert_allclose(_host(sim.state.qpos), [[.1], [9.]])
  np.testing.assert_allclose(_host(sim._applied_force), [[1.], [2.]])
  sim.restore(snap)
  np.testing.assert_allclose(_host(sim.state.qpos), [[.1], [.2]])
  np.testing.assert_allclose(_host(sim.state._qacc_warmstart), [[.7], [.8]])

  sim.copy_environment(0, 1)
  np.testing.assert_allclose(_host(sim._applied_force), [[1.], [1.]])
  sim.reset(env_ids=[1])
  np.testing.assert_allclose(_host(sim.state._qacc_warmstart), [[.7], [0.]])


def test_rejects_wrong_device_nonfinite_and_invalid_quaternion_before_write():
  import torch
  model = _model()
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  before = _host(sim.state.qpos)
  with pytest.raises(ValueError, match="finite"):
    mj_setState(sim, {"qpos": np.array([[np.inf]], np.float32)})
  with pytest.raises(ValueError, match="duplicates"):
    mj_setState(sim, {"qpos": np.array([[1.], [2.]], np.float32)}, env_ids=[0, 0])
  wrong_device = torch.tensor([[1.]], dtype=torch.float32, device="cpu")
  with pytest.raises(ValueError, match="must be on"):
    mj_setState(sim, {"qpos": wrong_device})
  np.testing.assert_array_equal(_host(sim.state.qpos), before)


def test_bool_eq_active_and_duplicate_aliases_are_handled_explicitly():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="100"/>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody><equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
    </mujoco>""")
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  mj_setState(sim, {"eq_active": np.array([True], dtype=np.bool_)})
  np.testing.assert_array_equal(_host(mj_getState(sim, StateSpec.EQ_ACTIVE)["eq_active"]), [[1]])
  with pytest.raises(ValueError, match="duplicate state field alias"):
    mj_setState(sim, {"warmstart": np.zeros((1, model.nv), np.float32),
                      "qacc_warmstart": np.ones((1, model.nv), np.float32)})
