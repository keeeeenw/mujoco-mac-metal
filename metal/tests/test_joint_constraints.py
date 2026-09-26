"""CPU oracle and explicit support-boundary checks for joint constraints."""
import numpy as np
import pytest
import mujoco

from mujoco_metal.joint_constraints import (
    JointConstraintDescriptor, joint_constraint_oracle, lower_joint_constraints,
)


XML = '''<mujoco><option timestep="0.002" gravity="0 0 0"><flag contact="disable"/></option>
<worldbody><body><joint name="a" type="hinge" axis="0 0 1" range="-0.2 0.2" limited="true" margin="0.03" frictionloss="0.3"/>
<geom type="capsule" size=".1 .2" mass="1"/><body pos="0 0 .4"><joint name="b" type="hinge" axis="0 1 0" range="-0.1 0.1" limited="true" margin="0.02" frictionloss="0.2"/>
<geom type="capsule" size=".1 .2" mass="1"/></body></body></worldbody>
<equality><joint name="link" joint1="a" joint2="b" polycoef="0 1 .2 0 0" solref="0.02 1"/></equality></mujoco>'''


def _mj_state(model, qpos, qvel):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  return data, mass


def test_lowering_is_immutable_and_preserves_row_capacity():
  model = mujoco.MjModel.from_xml_string(XML)
  d = lower_joint_constraints(model)
  assert isinstance(d, JointConstraintDescriptor)
  assert d.nrow == model.neq + model.nv + 2*model.njnt
  assert d.qpos0.flags.writeable is False
  with pytest.raises(ValueError):
    d.qpos0[0] = 0


def test_oracle_matches_real_mujoco_forward_for_coupled_equality_friction_limits():
  model = mujoco.MjModel.from_xml_string(XML)
  states = [([.24, .16], [1., -.4]), ([-.24, -.16], [-.7, .5]), ([.04, -.02], [.1, -.2])]
  expected = []
  masses, forces, qpos, qvel = [], [], [], []
  for p, v in states:
    data, mass = _mj_state(model, p, v)
    expected.append((data.qfrc_constraint.copy(), data.qacc.copy()))
    masses.append(mass); forces.append(data.qfrc_smooth.copy()); qpos.append(p); qvel.append(v)
  got = joint_constraint_oracle(model, np.array(qpos), np.array(qvel), np.array(masses), np.array(forces))
  assert np.all(got["status"] == 0)
  assert np.max(got["residual"]) <= 2e-6
  for i, (force, acc) in enumerate(expected):
    np.testing.assert_allclose(got["qfrc_constraint"][i], force, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got["qacc"][i], acc, rtol=2e-5, atol=2e-4)


def test_disable_and_per_batch_equality_activity_match_forward():
  model = mujoco.MjModel.from_xml_string(XML)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
  states = [([.24, .16], [1., -.4]), ([.24, .16], [1., -.4])]
  rows = [_mj_state(model, p, v) for p, v in states]
  output = joint_constraint_oracle(model, np.array([s[0] for s in states]),
      np.array([s[1] for s in states]), np.array([m for _, m in rows]),
      np.array([data.qfrc_smooth for data, _ in rows]),
      eq_active=np.array([[True], [False]]))
  for i, (data, _) in enumerate(rows):
    np.testing.assert_allclose(output["qfrc_constraint"][i], data.qfrc_constraint, rtol=2e-5, atol=2e-5)


def test_zero_dof_empty_world_is_valid_and_returns_empty_values():
  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
  output = joint_constraint_oracle(model, np.empty((2, 0)), np.empty((2, 0)),
                                  np.empty((2, 0, 0)), np.empty((2, 0)))
  assert output["qacc"].shape == (2, 0)
  assert output["qfrc_constraint"].shape == (2, 0)
  assert np.all(output["status"] == 0)


def test_unsupported_equality_family_fails_explicitly():
  connect = '''<mujoco><option><flag contact="disable"/></option><worldbody>
  <body name="a"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
  <body name="b" pos="1 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody>
  <equality><connect body1="a" body2="b" anchor="0 0 0"/></equality></mujoco>'''
  with pytest.raises(ValueError, match="only polynomial joint equality"):
    lower_joint_constraints(mujoco.MjModel.from_xml_string(connect))


def test_nonmatching_batches_and_nonfinite_values_rejected():
  model = mujoco.MjModel.from_xml_string(XML)
  with pytest.raises(ValueError, match="batch shapes"):
    joint_constraint_oracle(model, np.zeros((2, 2)), np.zeros((1, 2)),
                             np.zeros((2, 2, 2)), np.zeros((2, 2)))
  with pytest.raises(ValueError, match="finite"):
    joint_constraint_oracle(model, np.full((1, 2), np.nan), np.zeros((1, 2)),
                             np.eye(2)[None], np.zeros((1, 2)))


def test_metal_argument_abi_is_dense_and_matches_host_pack_order():
  import re
  from pathlib import Path
  source = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "joint_constraints.metal"
  indices = [int(x) for x in re.findall(r"\[\[buffer\((\d+)\)\]\]", source.read_text())]
  assert indices == list(range(30))
