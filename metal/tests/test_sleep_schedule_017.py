# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracle checks against pinned MuJoCo 3.10 sleep semantics."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.islands import build_island_partition
from mujoco_metal.sleep_schedule import (
    DeviceSleepScheduler,
    tree_sleep_transition_cpu,
)


def _model(body_attrs="sleep='allowed'", *, actuators=False, island="enable"):
  actuator = '<actuator><motor joint="j0"/></actuator>' if actuators else ""
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep="0.002" gravity="0 0 0"><flag sleep="enable" island="{island}"/></option>
    <worldbody>
      <body name="t0" {body_attrs}><joint name="j0" type="slide" axis="1 0 0"/><geom type="sphere" size="0.1"/></body>
      <body name="t1" pos="2 0 0" sleep="allowed"><joint name="j1" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1"/></body>
    </worldbody>{actuator}</mujoco>""")


def _transition(model, state, *, qvel=None, qfrc=None, xfrc=None,
                pairs=(), constraints=None, **wake):
  partition = build_island_partition(model)
  batch = state.shape[0]
  qvel = np.zeros((batch, model.nv), dtype=np.float32) if qvel is None else qvel
  qfrc = np.zeros_like(qvel) if qfrc is None else qfrc
  xfrc = np.zeros((batch, model.nbody, 6), dtype=np.float32) if xfrc is None else xfrc
  out = tree_sleep_transition_cpu(model, state, qvel, qfrc, xfrc,
                                  active_tree_pairs=pairs,
                                  constraints_active=constraints, **wake)
  return partition, out


def _asleep(model):
  """Pinned tree_asleep representation for independent singleton islands."""
  return np.arange(model.ntree, dtype=np.int32)[None, :]


def test_compiled_tree_partition_matches_pinned_mujoco_trees():
  model = _model(actuators=True)
  part = build_island_partition(model)
  np.testing.assert_array_equal(part.body_tree, model.body_treeid)
  np.testing.assert_array_equal(part.dof_tree, model.dof_treeid)
  assert part.ntree == model.ntree == 2
  assert part.tree_bodies == [
      list(range(int(model.tree_bodyadr[t]), int(model.tree_bodyadr[t] + model.tree_bodynum[t])))
      for t in range(model.ntree)]
  assert part.tree_dofs == [
      list(range(int(model.tree_dofadr[t]), int(model.tree_dofadr[t] + model.tree_dofnum[t])))
      for t in range(model.ntree)]
  assert model.tree_sleep_policy[0] == int(mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED)
  np.testing.assert_array_equal(part.tree_sleep_policy, model.tree_sleep_policy)


def test_sleep_transition_matches_ten_frame_pinned_awake_countdown():
  model = _model()
  state = np.full((1, model.ntree), -11, dtype=np.int32)
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)
  for _ in range(10):
    _, state = _transition(model, state)
    mujoco.mj_step(model, data)
  np.testing.assert_array_equal(state, data.tree_asleep.reshape(1, -1))
  np.testing.assert_array_equal(state, [[0, 1]])


def test_sleep_velocity_uses_dof_length_weighted_infinity_norm():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" sleep_tolerance="0.001"><flag sleep="enable"/></option>
    <worldbody><body sleep="allowed"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody>
</mujoco>""")


def _three_tree_model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option timestep="0.002" gravity="0 0 0"><flag sleep="enable"/></option>
    <worldbody>
      <body name="a" sleep="allowed"><joint type="slide"/><geom size=".1"/></body>
      <body name="b" pos="2 0 0" sleep="allowed"><joint type="slide"/><geom size=".1"/></body>
      <body name="c" pos="4 0 0" sleep="allowed"><joint type="slide"/><geom size=".1"/></body>
    </worldbody></mujoco>""")
  state = np.asarray([[-11]], dtype=np.int32)
  qvel = np.zeros((1, model.nv), dtype=np.float32)
  qvel[0, 3] = 0.005  # angular dof length is 0.1, so weighted speed is 0.0005
  _, out = _transition(model, state, qvel=qvel)
  assert out[0, 0] == -10
  qvel[0, 0] = 0.002  # translational length 1.0 exceeds tolerance
  _, out = _transition(model, state, qvel=qvel)
  assert out[0, 0] == -11


@pytest.mark.parametrize("force", ["qfrc", "xfrc"])
def test_applied_force_wakes_only_its_own_independent_tree(force):
  model = _model()
  part = build_island_partition(model)
  state = _asleep(model)
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  xfrc = np.zeros((1, model.nbody, 6), dtype=np.float32)
  if force == "qfrc":
    qfrc[0, part.tree_dofs[0][0]] = 1.0
  else:
    xfrc[0, part.tree_bodies[0][0], 4] = 1.0
  _, out = _transition(model, state, qfrc=qfrc, xfrc=xfrc)
  np.testing.assert_array_equal(out, [[-11, 1]])


def test_applied_force_and_active_link_propagate_wake_association():
  model = _model(actuators=True)
  state = _asleep(model)
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, int(model.tree_dofadr[0])] = 1.0
  _, out = _transition(model, state, qfrc=qfrc)
  np.testing.assert_array_equal(out, [[-11, 1]])
  qfrc.fill(0)
  _, out = _transition(model, out, qfrc=qfrc, pairs=[(0, 1)])
  np.testing.assert_array_equal(out, [[-10, -10]])


def test_waking_sleeping_constraint_island_matches_mujoco_cycle_state():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0"><flag sleep="enable"/></option>
    <worldbody>
      <body name="a" sleep="allowed"><freejoint/><geom size="0.1" mass="1"/></body>
      <body name="b" pos="0.5 0 0" sleep="allowed"><freejoint/><geom size="0.1" mass="1"/></body>
    </worldbody>
    <equality><weld body1="a" body2="b" active="true"/></equality>
  </mujoco>""")
  data = mujoco.MjData(model)
  for _ in range(12):
    mujoco.mj_step(model, data)
  assert np.all(data.tree_asleep >= 0)
  assert tuple(data.tree_asleep) in ((0, 1), (1, 0))
  qvel = data.qvel[None, :].copy()
  qfrc = np.zeros_like(qvel)
  qfrc[0, int(model.tree_dofadr[0])] = 1.0
  xfrc = np.zeros((1, model.nbody, 6), dtype=np.float32)
  expected = tree_sleep_transition_cpu(
      model, data.tree_asleep[None, :], qvel, qfrc, xfrc)
  data.qfrc_applied[:] = qfrc[0]
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(expected[0], data.tree_asleep)


def test_pairwise_equality_wakes_two_distinct_asleep_cycles_fully():
  model = _model()
  _, result = _transition(
      model, _asleep(model), pairs=(), active_equality_pairs=[[0, 1]])
  np.testing.assert_array_equal(result, [[-10, -10]])


def test_pairwise_equality_wakes_sleeping_cycle_to_fully_awake_value():
  model = _model()
  _, result = _transition(
      model, np.array([[-5, 1]], dtype=np.int32),
      active_equality_pairs=[[0, 1]])
  # The awake cycle advances from -5 to -4; the sleeping cycle receives the
  # pinned equality kAwake=-11 and then advances independently to -10.
  np.testing.assert_array_equal(result, [[-4, -10]])


def test_flex_equality_hyperedge_uses_first_awake_and_first_sleeping_tree():
  model = _three_tree_model()
  _, result = _transition(
      model, np.array([[-5, 1, 2]], dtype=np.int32),
      active_flex_equalities=[[0, 1, 2]])
  # mj_wakeEquality passes the selected awake tree's current countdown (-5),
  # waking just the first sleeping island in the flex's body order.
  np.testing.assert_array_equal(result, [[-4, -4, 2]])


def test_flex_equality_does_not_wake_disconnected_sleepers_without_awake_member():
  model = _three_tree_model()
  _, result = _transition(
      model, np.array([[0, 1, 2]], dtype=np.int32),
      active_flex_equalities=[[0, 1, 2]])
  np.testing.assert_array_equal(result, [[0, 1, 2]])


def test_never_and_auto_never_policies_refuse_sleep():
  never = _model("sleep='never'")
  _, out = _transition(never, _asleep(never))
  assert out[0, 0] == -11 and out[0, 1] == 1
  automatic = _model("sleep='auto'", actuators=True)
  assert automatic.tree_sleep_policy[0] == int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER)
  _, out = _transition(automatic, _asleep(automatic))
  assert out[0, 0] == -11


def test_disabled_island_path_cannot_sleep_a_constrained_world():
  model = _model(island="disable")
  state = _asleep(model)
  _, out = _transition(model, state, constraints=np.asarray([True]))
  np.testing.assert_array_equal(out, [[-11, -11]])


def test_simulation_sleep_wake_runs_after_current_fk_and_narrowphase():
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  events = []

  class FK:
    def run_device(self, qpos, mpos, mquat, *, tree_awake):
      assert tree_awake.tolist() == [[1, 1]]
      events.append("fk")
      return {"geom_pos": torch.zeros((1, 1, 3)), "geom_quat": torch.zeros((1, 1, 4))}

  class Coupled:
    descriptor = type("Descriptor", (), {"neq": 1, "npairs": 1})()
    _empty_slot_map = object()
    _workspace = {}

    def generate_candidates(self, poses, qvel, eq_active):
      events.append("narrowphase")
      self._workspace["slot_maps"] = type(
          "SlotMap", (), {"logical_to_packed": torch.tensor([[0]], dtype=torch.int32)})()
      self._workspace["active_tree_links"] = torch.tensor(
          [[[0, 1]]], dtype=torch.int32)
      self._workspace["active_tree_link_overflow"] = torch.zeros((1,), dtype=torch.int32)

  class Scheduler:
    tree_awake = torch.ones((1, 2), dtype=torch.int32)

    def wake_before_solve(self, qvel, **kwargs):
      assert kwargs["active_tree_pairs"].tolist() == [[[0, 1]]]
      events.append("wake")

  coupled = Coupled()
  sim = MetalSimulation.__new__(MetalSimulation)
  sim._state = type("State", (), {
      "_torch": torch, "_mpos": None, "_mquat": None,
      "_eq_active": torch.ones((1, 1), dtype=torch.int32)})()
  sim._smooth = type("Smooth", (), {"_fk": FK()})()
  sim._coupled_constraints = coupled
  sim._active_contact_link_workspace = None
  sim._sleep_schedule = Scheduler()
  sim._applied_force = None
  sim._body_wrench = None
  sim.batch_size = 1
  sim._prepare_sleep_schedule(torch.zeros((1, 1)), torch.zeros((1, 1)))
  assert events == ["fk", "narrowphase", "wake"]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_device_scheduler_keeps_batch_local_force_and_contact_wakes():
  import torch
  model = _model()
  scheduler = DeviceSleepScheduler(model, batch_size=2, max_tree_links=2)
  qvel = torch.zeros((2, model.nv), dtype=torch.float32, device="mps")
  qfrc = torch.zeros_like(qvel)
  xfrc = torch.zeros((2, model.nbody, 6), dtype=torch.float32, device="mps")
  for _ in range(10):
    awake = scheduler.step(qvel, qfrc_applied=qfrc, xfrc_applied=xfrc)
  # The independently compiled pinned model enters its INIT tree at 0 and
  # the second allowed tree at 1 after ten advance passes.
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[0, 1], [0, 1]])
  qfrc[1, int(model.tree_dofadr[0])] = 1.0
  awake = scheduler.step(qvel, qfrc_applied=qfrc, xfrc_applied=xfrc)
  np.testing.assert_array_equal(awake.cpu().numpy(), [[0, 0], [1, 0]])
  qfrc.zero_()
  links = torch.full((2, 2, 2), -1, dtype=torch.int32, device="mps")
  links[0, 0] = torch.tensor((0, 1), dtype=torch.int32, device="mps")
  awake = scheduler.step(qvel, qfrc_applied=qfrc, xfrc_applied=xfrc,
                         active_tree_pairs=links)
  np.testing.assert_array_equal(awake.cpu().numpy(), [[1, 1], [1, 0]])
