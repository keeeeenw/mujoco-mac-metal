# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracle checks against pinned MuJoCo 3.10 sleep semantics."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.islands import build_island_partition
from mujoco_metal.sleep_schedule import (
    awake_lists_cpu,
    DeviceSleepScheduler,
    sleep_scheduler_workspace_sizes,
    tree_sleep_transition_cpu,
    validate_sleep_equalities,
    initial_sleep_state,
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


def test_multitree_tendon_pinned_policy_keeps_auto_awake_and_rejects_allowed():
  # engine_setconst.c marks every AUTO tree touched by a tendon spanning more
  # than two trees as AUTO_NEVER. Explicit ALLOWED/INIT sleep is rejected.
  links = "".join(
      f'<joint joint="j{i}" coef="1"/>' for i in range(3))
  tree_bodies = "".join(
      f'<body name="b{i}" pos="{2 * i} 0 0" sleep="auto">'
      f'<joint name="j{i}" type="slide"/><geom size=".1"/></body>'
      for i in range(3))
  auto_xml = ("<mujoco><option><flag sleep=\"enable\"/></option>"
              f"<worldbody>{tree_bodies}</worldbody><tendon>"
              f"<fixed name=\"three_tree\" limited=\"false\">{links}"
              "</fixed></tendon></mujoco>")
  model = mujoco.MjModel.from_xml_string(auto_xml)
  assert int(model.tendon_treenum[0]) == 3
  np.testing.assert_array_equal(
      model.tree_sleep_policy,
      np.full(model.ntree,
              int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER), dtype=np.int32))

  allowed_xml = auto_xml.replace('sleep="auto"', 'sleep="allowed"')
  with pytest.raises(ValueError, match="spans more than 2 trees, sleeping not allowed"):
    mujoco.MjModel.from_xml_string(allowed_xml)


def test_awake_list_lowering_matches_pinned_static_body_and_dof_rules_cpu():
  model = _model()
  # Worlds: both dynamic trees asleep; only tree 0 awake; only tree 1 awake.
  masks = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
  lists = awake_lists_cpu(model, masks)
  for world, awake in enumerate(masks):
    body_tree = np.asarray(model.body_treeid)
    parent = np.asarray(model.body_parentid)
    dof_tree = np.asarray(model.dof_treeid)
    expected_body = [b for b in range(model.nbody)
                     if body_tree[b] < 0 or awake[body_tree[b]]]
    expected_parent = [b for b in range(1, model.nbody)
                       if body_tree[parent[b]] < 0 or awake[body_tree[parent[b]]]]
    expected_dof = [d for d in range(model.nv)
                    if dof_tree[d] >= 0 and awake[dof_tree[d]]]
    bc, pc, dc = (int(lists[k][world]) for k in
                  ("body_count", "parent_count", "dof_count"))
    np.testing.assert_array_equal(lists["body_ids"][world, :bc], expected_body)
    np.testing.assert_array_equal(lists["parent_ids"][world, :pc], expected_parent)
    np.testing.assert_array_equal(lists["dof_ids"][world, :dc], expected_dof)
    assert np.all(lists["body_ids"][world, bc:] == -1)
    assert np.all(lists["dof_ids"][world, dc:] == -1)
  # The world body is static and always participates, but static world DOFs
  # do not exist; each selected dynamic tree contributes only its own DOFs.
  assert lists["body_ids"][0, 0] == 0
  assert lists["dof_count"].tolist() == [0, 1, 1]


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


def _initial_sleep_model(connected):
  model = _model(body_attrs="sleep='init'")
  if connected:
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="0 0 0"><flag sleep="enable" contact="disable"/></option>
      <worldbody>
        <body name="a" sleep="init"><joint type="slide"/><geom size=".1"/></body>
        <body name="b" pos="2 0 0" sleep="init"><joint type="slide"/><geom size=".1"/></body>
        <body name="c" pos="4 0 0" sleep="allowed"><joint type="slide"/><geom size=".1"/></body>
      </worldbody><equality><connect body1="a" body2="b" anchor="1 0 0"/></equality>
      </mujoco>''')
  return model


@pytest.mark.parametrize("connected", [False, True])
def test_initial_sleep_template_matches_pinned_reset_and_is_immutable(connected):
  model = _initial_sleep_model(connected)
  expected = mujoco.MjData(model).tree_asleep.copy()
  template = initial_sleep_state(model)
  np.testing.assert_array_equal(template, expected)
  assert template[-1] == -10
  if connected:
    np.testing.assert_array_equal(template[:2], [1, 0])
  assert not template.flags.writeable
  with pytest.raises(ValueError):
    template[0] = -11


def test_pinned_kinematics_wakes_init_sleep_cycle_on_retained_pose_mismatch():
  """mj_kinematics1 compares candidate joint-body xpose to reset xpose."""
  # Resolve the sibling fixture from this test source, including when the
  # package is qualified from an immutable snapshot outside the checkout.
  import importlib.util
  from pathlib import Path
  fixture_path = Path(__file__).with_name("test_integrated_scalable_public_017.py")
  spec = importlib.util.spec_from_file_location("scalable_sleep_fixture", fixture_path)
  fixture = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(fixture)
  _XML = fixture._XML
  xml = (_XML.replace('<option timestep=',
                      '<option integrator="RK4" timestep=')
         .replace('contact="disable"',
                  'contact="disable" sleep="enable"')
         .replace('<body name="left"', '<body name="left" sleep="init"')
         .replace('<body name="right"', '<body name="right" sleep="init"'))
  model = mujoco.MjModel.from_xml_string(xml)
  unchanged = mujoco.MjData(model)
  reset_pose = unchanged.xpos.copy()
  mujoco.mj_forward(model, unchanged)
  np.testing.assert_array_equal(unchanged.tree_asleep, [1, 0])
  np.testing.assert_array_equal(unchanged.xpos, reset_pose)

  moved = mujoco.MjData(model)
  moved.qpos[:] = [.03, -.01]
  mujoco.mj_forward(model, moved)
  np.testing.assert_array_equal(moved.tree_asleep, [-11, -11])
  np.testing.assert_array_equal(moved.tree_awake, [1, 1])
  np.testing.assert_array_equal(moved.xpos[1:, 0], [-.97, .99])


def test_sleep_workspace_inventory_is_exact_and_rejects_before_torch(monkeypatch):
  import builtins
  model = _model()
  sizes = sleep_scheduler_workspace_sizes(model, 3, 7)
  assert sizes["sleep.initial_tree_state"] == max(model.ntree, 1)
  assert sizes["sleep.tree_state"] == 3 * max(model.ntree, 1)
  assert sizes["sleep.contact_equality_links"] == 3 * 7 * 2
  assert sizes["sleep.zero_xfrc"] == 3 * model.nbody * 6
  assert sizes["sleep.dof_length"] == model.nv
  assert sizes["sleep.awake_counts"] == 3 * 3

  original_import = builtins.__import__
  def guarded_import(name, *args, **kwargs):
    if name == "torch":
      raise AssertionError("torch imported before scheduler admission")
    return original_import(name, *args, **kwargs)
  monkeypatch.setattr(builtins, "__import__", guarded_import)
  with pytest.raises(ValueError, match="batch_size exceeds.*int32"):
    DeviceSleepScheduler(model, batch_size=1 << 31)
  with pytest.raises(ValueError, match="max_tree_links.*integer"):
    DeviceSleepScheduler(model, batch_size=1, max_tree_links=2.5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("connected", [False, True])
def test_native_initial_sleep_and_selected_reset_match_pinned_template(connected):
  model = _initial_sleep_model(connected)
  scheduler = DeviceSleepScheduler(model, 2)
  expected = mujoco.MjData(model).tree_asleep.copy()
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(),
                                np.broadcast_to(expected, (2, model.ntree)))
  scheduler.tree_state.fill_(-7)
  scheduler.tree_awake.fill_(1)
  scheduler.links.fill_(0)
  scheduler.constraints_active.fill_(1)
  scheduler.links_overflow.fill_(1)
  scheduler.status.fill_(3)
  scheduler.eq_active.zero_()
  scheduler._build_awake_lists()
  fields = ("tree_state", "tree_awake", "links", "constraints_active",
            "links_overflow", "status", "eq_active", "awake_counts",
            "body_awake_ids", "parent_awake_ids", "body_awake_mask",
            "dof_awake_ids")
  before = {name: getattr(scheduler, name)[1].cpu().numpy().copy() for name in fields}
  initial = scheduler._initial_tree_state.cpu().numpy().copy()
  default_eq = scheduler.eq_active_default[1].cpu().numpy().copy()
  scheduler.reset([0])
  np.testing.assert_array_equal(scheduler.tree_state[0].cpu().numpy(), expected)
  np.testing.assert_array_equal(scheduler._initial_tree_state.cpu().numpy(), initial)
  np.testing.assert_array_equal(scheduler.eq_active_default[1].cpu().numpy(), default_eq)
  for name in fields:
    np.testing.assert_array_equal(getattr(scheduler, name)[1].cpu().numpy(), before[name])


def test_cpu_failed_world_sleep_state_is_untouched():
  model = _model()
  initial = np.array([[0, 1], [-11, -11]], dtype=np.int32)
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  qvel[0, int(model.tree_dofadr[0])] = 10.0
  qfrc = np.zeros_like(qvel)
  xfrc = np.zeros((2, model.nbody, 6), dtype=np.float32)
  result = tree_sleep_transition_cpu(
      model, initial, qvel, qfrc, xfrc, active_worlds=np.array([0, 1]))
  np.testing.assert_array_equal(result[0], initial[0])
  np.testing.assert_array_equal(result[1], [-10, -10])
  with pytest.raises(ValueError, match="active_worlds"):
    tree_sleep_transition_cpu(
        model, initial, qvel, qfrc, xfrc, active_worlds=np.array([0, 2]))


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


def test_cpu_sleep_wake_uses_exact_zero_velocity_not_sleep_tolerance():
  """mj_wake's zero-tolerance check is distinct from mj_sleep's threshold."""
  model = _model()
  model.opt.sleep_tolerance = 0.1
  data = mujoco.MjData(model)
  data.tree_asleep[:] = [0, 1]  # two valid singleton sleeping cycles
  data.qvel[:] = [0.05, 0.01]  # both speeds are below sleep_tolerance
  mujoco.mj_forward(model, data)
  np.testing.assert_array_equal(data.tree_asleep, [-11, -11])

  # The CPU scheduler oracle combines pre-solve wake and post-step sleep:
  # it may count the newly woken quiet trees down to -10, but must not leave
  # either one in a nonnegative sleeping state.
  _, transitioned = _transition(
      model, np.asarray([[0, 1]], dtype=np.int32),
      qvel=np.asarray([[0.05, 0.01]], dtype=np.float32))
  assert np.all(transitioned < 0), transitioned


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


def test_current_contact_between_sleeping_trees_is_pinned_invariant_error():
  model = _model()
  with pytest.raises(ValueError, match="active contact joins sleeping trees"):
    _transition(model, _asleep(model), pairs=[(0, 1)])


def test_tendon_equality_sleep_boundary_fails_explicitly():
  class TendonEqualityModel:
    neq = 1
    nflex = 0
    body_treeid = np.array([-1, 0], dtype=np.int32)
    eq_type = np.array([mujoco.mjtEq.mjEQ_TENDON], dtype=np.int32)
    eq_objtype = np.array([mujoco.mjtObj.mjOBJ_TENDON], dtype=np.int32)
    eq_obj1id = np.array([0], dtype=np.int32)
    eq_obj2id = np.array([1], dtype=np.int32)
    site_bodyid = np.zeros((0,), dtype=np.int32)
    jnt_bodyid = np.zeros((0,), dtype=np.int32)
    flex_interp = np.zeros((0,), dtype=np.int32)
    flex_nodeadr = np.zeros((0,), dtype=np.int32)
    flex_nodenum = np.zeros((0,), dtype=np.int32)
    flex_nodebodyid = np.zeros((0,), dtype=np.int32)
    flex_vertadr = np.zeros((0,), dtype=np.int32)
    flex_vertnum = np.zeros((0,), dtype=np.int32)
    flex_vertbodyid = np.zeros((0,), dtype=np.int32)

  with pytest.raises(NotImplementedError, match="does not support sleeping for tendon"):
    validate_sleep_equalities(TendonEqualityModel(), enabled=True)
  plan = validate_sleep_equalities(TendonEqualityModel(), enabled=False)
  np.testing.assert_array_equal(plan["unsupported_tendon_eq_ids"], [0])


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


def test_sleep_link_input_accepts_bounded_dynamic_axis_and_never_truncates():
  torch = pytest.importorskip("torch")
  from mujoco_metal.sleep_schedule import _tree_link_count

  class LinkTensor:
    def __init__(self, shape, device="mps:0", dtype=None, contiguous=True):
      self.shape = shape
      self.device = torch.device(device)
      self.dtype = torch.int32 if dtype is None else dtype
      self._contiguous = contiguous

    def is_contiguous(self):
      return self._contiguous

  # The unindexed default MPS device accepts the backend's mps:0 tensor view.
  assert _tree_link_count(
      LinkTensor((2, 3, 2)), 2, 5, torch.device("mps"), torch) == 3
  with pytest.raises(ValueError, match="links <= 5"):
    _tree_link_count(LinkTensor((2, 6, 2)), 2, 5,
                     torch.device("mps"), torch)
  with pytest.raises(ValueError, match="batch, links, 2"):
    _tree_link_count(LinkTensor((2, 3, 4)), 2, 5,
                     torch.device("mps"), torch)
  with pytest.raises(ValueError, match="on mps:1"):
    _tree_link_count(LinkTensor((2, 3, 2), device="mps:0"), 2, 5,
                     torch.device("mps:1"), torch)
  with pytest.raises(ValueError, match="int32"):
    _tree_link_count(LinkTensor((2, 3, 2), dtype=torch.float32), 2, 5,
                     torch.device("mps"), torch)
  with pytest.raises(ValueError, match="contiguous"):
    _tree_link_count(LinkTensor((2, 3, 2), contiguous=False), 2, 5,
                     torch.device("mps"), torch)
  with pytest.raises(ValueError, match="active_tree_pairs must be on mps"):
    _tree_link_count(torch.zeros((2, 3, 2), dtype=torch.int32), 2, 5,
                     torch.device("mps"), torch)


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

    def pose_mismatch(self):
      return torch.zeros((1, 2), dtype=torch.int32)

  class Coupled:
    descriptor = type("Descriptor", (), {"neq": 1, "npairs": 1})()
    _empty_slot_map = object()
    _workspace = {}

    def generate_candidates(self, poses, qvel, eq_active=None, **kwargs):
      events.append("narrowphase")
      self._workspace["slot_maps"] = type(
          "SlotMap", (), {"logical_to_packed": torch.tensor([[0]], dtype=torch.int32)})()
      self._workspace["active_tree_links"] = torch.tensor(
          [[[0, 1]]], dtype=torch.int32)
      self._workspace["active_tree_link_overflow"] = torch.zeros((1,), dtype=torch.int32)

  class Scheduler:
    tree_awake = torch.ones((1, 2), dtype=torch.int32)

    def wake_before_solve(self, qvel, **kwargs):
      if "active_tree_pairs" not in kwargs:
        assert kwargs["pose_mismatch"].tolist() == [[0, 0]]
        events.append("pose_wake")
        return
      assert kwargs["active_tree_pairs"].tolist() == [[[0, 1]]]
      assert kwargs["equality_active"].tolist() == [[1]]
      events.append("wake")

  coupled = Coupled()
  sim = MetalSimulation.__new__(MetalSimulation)
  sim._state = type("State", (), {
      "_torch": torch, "_mpos": None, "_mquat": None,
      "_eq_active": torch.ones((1, 1), dtype=torch.int32),
      "_status": torch.zeros((1,), dtype=torch.int32)})()
  sim._smooth = type("Smooth", (), {"_fk": FK()})()
  sim._flex = None
  sim._coupled_constraints = coupled
  sim._active_contact_link_workspace = None
  sim._sleep_schedule = Scheduler()
  sim._applied_force = None
  sim._body_wrench = None
  sim.batch_size = 1
  sim._prepare_sleep_schedule(torch.zeros((1, 1)), torch.zeros((1, 1)))
  assert events == ["fk", "pose_wake", "fk", "narrowphase", "wake"]


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
  # A valid active contact has one awake and one sleeping tree; MuJoCo's
  # contact wake copies the awake partner countdown into the sleeper.
  scheduler.tree_state[0] = torch.tensor([-5, 1], dtype=torch.int32, device="mps")
  links = torch.full((2, 2, 2), -1, dtype=torch.int32, device="mps")
  links[0, 0] = torch.tensor((0, 1), dtype=torch.int32, device="mps")
  awake = scheduler.step(qvel, qfrc_applied=qfrc, xfrc_applied=xfrc,
                         active_tree_pairs=links)
  np.testing.assert_array_equal(awake.cpu().numpy(), [[1, 1], [1, 0]])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_device_failed_world_preserves_scheduler_state_and_counters():
  import torch
  model = _model()
  scheduler = DeviceSleepScheduler(model, batch_size=2, max_tree_links=1)
  device = torch.device("mps")
  # World zero stands in for a failed solve row. Deliberately seed non-policy
  # values in every scheduler-owned output so a kernel that writes even a
  # recomputed equivalent value is distinguishable from the required early
  # return. World one remains eligible for the ordinary pre-integration pass.
  scheduler.tree_state.copy_(torch.tensor([[31, 32], [-11, -11]], dtype=torch.int32, device=device))
  scheduler.tree_awake.copy_(torch.tensor([[7, 8], [1, 1]], dtype=torch.int32, device=device))
  scheduler.status.copy_(torch.tensor([91, 0], dtype=torch.int32, device=device))
  scheduler.awake_counts.copy_(torch.tensor([[41, 42, 43], [0, 0, 0]], dtype=torch.int32, device=device))
  scheduler.body_awake_ids.fill_(101)
  scheduler.parent_awake_ids.fill_(102)
  scheduler.dof_awake_ids.fill_(103)
  scheduler.body_awake_mask.fill_(104)
  failed_before = [tensor[0].clone() for tensor in (
      scheduler.tree_state, scheduler.tree_awake, scheduler.status,
      scheduler.awake_counts, scheduler.body_awake_ids,
      scheduler.parent_awake_ids, scheduler.dof_awake_ids,
      scheduler.body_awake_mask)]
  qvel = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  qfrc = torch.zeros_like(qvel)
  xfrc = torch.zeros((2, model.nbody, 6), dtype=torch.float32, device=device)
  active_worlds = torch.tensor([False, True], dtype=torch.bool, device=device)
  scheduler.advance_before_integration(
      qvel, qfrc_applied=qfrc, xfrc_applied=xfrc,
      active_worlds=active_worlds)
  failed_after = [tensor[0] for tensor in (
      scheduler.tree_state, scheduler.tree_awake, scheduler.status,
      scheduler.awake_counts, scheduler.body_awake_ids,
      scheduler.parent_awake_ids, scheduler.dof_awake_ids,
      scheduler.body_awake_mask)]
  for before, after in zip(failed_before, failed_after):
    torch.testing.assert_close(after, before, rtol=0, atol=0)
  np.testing.assert_array_equal(scheduler.tree_state[1].cpu().numpy(), [-10, -10])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_device_contact_between_sleeping_trees_sets_pinned_invariant_status():
  import torch
  model = _model()
  scheduler = DeviceSleepScheduler(model, batch_size=2, max_tree_links=1)
  scheduler.tree_state.copy_(torch.tensor([[0, 1], [-11, -11]], dtype=torch.int32, device="mps"))
  qvel = torch.zeros((2, model.nv), dtype=torch.float32, device="mps")
  qfrc = torch.zeros_like(qvel)
  xfrc = torch.zeros((2, model.nbody, 6), dtype=torch.float32, device="mps")
  links = torch.tensor([[[0, 1]], [[-1, -1]]], dtype=torch.int32, device="mps")
  scheduler.wake_before_solve(
      qvel, qfrc_applied=qfrc, xfrc_applied=xfrc,
      active_tree_pairs=links)
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[0, 1], [-11, -11]])
  np.testing.assert_array_equal(scheduler.tree_awake.cpu().numpy(), [[0, 0], [1, 1]])
  np.testing.assert_array_equal(scheduler.status.cpu().numpy(), [3, 0])
  with pytest.raises(RuntimeError, match="contact between sleeping trees"):
    scheduler.check_status()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_device_equality_wake_is_separate_and_model_ordered():
  import torch
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0"><flag sleep="enable"/></option>
    <worldbody>
      <body name="a" sleep="allowed"><freejoint/><geom size=".1"/></body>
      <body name="b" pos="1 0 0" sleep="allowed"><freejoint/><geom size=".1"/></body>
    </worldbody>
    <equality><weld body1="a" body2="b" active="true"/></equality>
  </mujoco>""")
  scheduler = DeviceSleepScheduler(model, batch_size=1, max_tree_links=1)
  qvel = torch.zeros((1, model.nv), dtype=torch.float32, device="mps")
  qfrc = torch.zeros_like(qvel)
  xfrc = torch.zeros((1, model.nbody, 6), dtype=torch.float32, device="mps")
  scheduler.tree_state.copy_(torch.tensor([[-5, 1]], dtype=torch.int32, device="mps"))
  awake = scheduler.wake_before_solve(
      qvel, qfrc_applied=qfrc, xfrc_applied=xfrc,
      active_tree_pairs=torch.full((1, 1, 2), -1, dtype=torch.int32, device="mps"),
      equality_active=torch.ones((1, model.neq), dtype=torch.int32, device="mps"))
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[-5, -11]])
  np.testing.assert_array_equal(awake.cpu().numpy(), [[1, 1]])
  scheduler.tree_state.copy_(torch.tensor([[-5, 1]], dtype=torch.int32, device="mps"))
  inactive = torch.zeros((1, model.neq), dtype=torch.int32, device="mps")
  awake = scheduler.wake_before_solve(
      qvel, qfrc_applied=qfrc, xfrc_applied=xfrc,
      active_tree_pairs=torch.full((1, 1, 2), -1, dtype=torch.int32, device="mps"),
      equality_active=inactive)
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[-5, 1]])
  np.testing.assert_array_equal(awake.cpu().numpy(), [[1, 0]])
