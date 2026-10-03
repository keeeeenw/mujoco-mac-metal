# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Nonzero inverse constraint and reusable split-stage regressions."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal.native_api import (
    mj_fwdAcceleration,
    mj_fwdPosition,
    mj_fwdVelocity,
    mj_inverse,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
]


def _host(value):
  return value.detach().cpu().numpy().copy()


def _equality_model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="100"/>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/><geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/><geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody><equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
  </mujoco>""")


@pytest.mark.parametrize("offset", [0.02, -0.02])
def test_inverse_bilateral_equality_retains_both_force_signs(offset):
  model = _equality_model()
  data = mujoco.MjData(model)
  data.qpos[:] = [offset, 0]
  mujoco.mj_inverse(model, data)
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
                        profile="integrated_euler_v1")
  result = _host(mj_inverse(sim))[0]
  np.testing.assert_allclose(result, data.qfrc_inverse, atol=2e-3, rtol=2e-3)
  assert np.linalg.norm(data.qfrc_constraint) > 1
  assert np.sign(data.qfrc_constraint[0]) == -np.sign(offset)


def test_split_stages_reuse_supplied_pose_dynamics_and_force_inputs():
  model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0"/>
    <worldbody><body><joint name="a" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size=".08" mass="1" contype="0" conaffinity="0"/>
      <body pos="1 0 0"><joint name="b" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".06" mass=".5" contype="0" conaffinity="0"/>
      </body>
    </body></worldbody></mujoco>""")
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  qp = np.array([[.3, -.4]], np.float32)
  qv = np.array([[.4, -.2]], np.float32)
  poses = mj_fwdPosition(sim, qp)
  # If velocity recomputes FK, this raises. It must consume the exact pose
  # workspace produced by the prior stage, including nontrivial articulated
  # configurations where the dense inertia depends on qpos.
  sim._smooth._fk.run_device = lambda *a, **k: (_ for _ in ()).throw(AssertionError("FK reran"))
  dynamics = mj_fwdVelocity(sim, qp, qv, poses=poses)
  assert dynamics["poses"] is poses
  data = mujoco.MjData(model)
  data.qpos[:] = qp[0]
  data.qvel[:] = qv[0]
  mujoco.mj_forward(model, data)
  expected_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, expected_mass)
  np.testing.assert_allclose(_host(dynamics["mass_matrix"])[0], expected_mass,
                             atol=2e-5, rtol=2e-5)
  acceleration, status = mj_fwdAcceleration(
      sim, qp, qv, poses=poses, dynamics=dynamics,
      qfrc_applied=np.array([[2., 0.]], np.float32))
  expected_acc = np.linalg.solve(
      expected_mass, np.array([2., 0.]) - _host(dynamics["qfrc_bias"])[0])
  np.testing.assert_allclose(_host(acceleration)[0], expected_acc, atol=2e-5, rtol=2e-5)
  np.testing.assert_array_equal(_host(status), [0])
  np.testing.assert_allclose(_host(sim.state.qpos), [[0, 0]], atol=0)
  np.testing.assert_allclose(_host(sim._applied_force), [[0, 0]], atol=0)


def test_inverse_includes_armature_mass_and_preserves_solver_cache():
  model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0" solver="PGS" iterations="100"/>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0" armature="2"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>""")
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  data = mujoco.MjData(model)
  data.qacc[:] = 1
  mujoco.mj_inverse(model, data)
  cache_valid = sim._assembled_system_valid
  result = _host(mj_inverse(sim, qacc=np.ones((1, model.nv), dtype=np.float32)))[0]
  np.testing.assert_allclose(result, data.qfrc_inverse, atol=1e-4, rtol=1e-4)
  assert sim._assembled_system_valid == cache_valid


@pytest.mark.parametrize("stepped", [False, True])
@pytest.mark.parametrize("fail_gradient", [False, True])
def test_inverse_preserves_structured_contact_workspace_and_next_step(
    stepped, fail_gradient, monkeypatch):
  """Queries and late errors preserve canonical map storage and replay."""
  import dataclasses
  import torch

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" solver="PGS" iterations="100"/>
    <worldbody><geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"
      mass="1" condim="3"/></body></worldbody></mujoco>''')
  qpos = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  qpos[1, 2] += .4
  sim = MetalSimulation(model, 2, qpos=qpos, profile="integrated_euler_v1")
  if stepped:
    sim.step()
  checkpoint = sim.snapshot()
  control = MetalSimulation(model, 2, profile="integrated_euler_v1")
  control.restore(checkpoint)
  workspace = sim._coupled_constraints._workspace

  def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
      yield prefix, value
    elif isinstance(value, dict):
      for key, item in value.items():
        yield from tensors(item, prefix + "/" + key)
    elif dataclasses.is_dataclass(value):
      for field in dataclasses.fields(value):
        yield from tensors(getattr(value, field.name), prefix + "/" + field.name)

  original_keys = set(workspace)
  saved = {key: (_host(value), value.data_ptr()) for key, value in tensors(workspace)}
  qacc = torch.ones((2, model.nv), dtype=torch.float32, device="mps")
  if fail_gradient:
    original_bmm = torch.bmm
    nr = sim._coupled_constraints.descriptor.nr

    def fail_after_constraint_assembly(left, right, *args, **kwargs):
      if left.shape[-2:] == (nr, model.nv):
        raise RuntimeError("inverse row-gradient failure")
      return original_bmm(left, right, *args, **kwargs)

    with monkeypatch.context() as patch:
      patch.setattr(torch, "bmm", fail_after_constraint_assembly)
      with pytest.raises(RuntimeError, match="inverse row-gradient failure"):
        mj_inverse(sim, qacc=qacc)
  else:
    mj_inverse(sim, qacc=qacc)
  assert set(workspace) == original_keys
  current = dict(tensors(workspace))
  assert set(current) == set(saved)
  for key, (expected, pointer) in saved.items():
    np.testing.assert_array_equal(_host(current[key]), expected, err_msg=key)
    assert current[key].data_ptr() == pointer, key
  sim.step()
  control.step()
  for field in ("qpos", "qvel", "qacc", "time", "status", "qacc_warmstart"):
    np.testing.assert_array_equal(_host(getattr(sim.state, "_" + field)),
                                  _host(getattr(control.state, "_" + field)))


@pytest.mark.parametrize("cone,condim", [
    ("elliptic", 3), ("elliptic", 4), ("elliptic", 6),
    ("pyramidal", 3), ("pyramidal", 4), ("pyramidal", 6),
])
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_inverse_contact_friction_cone_matches_pinned_force_gradient(cone, condim, solver):
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 -9.81" cone="{cone}" solver="{solver}" iterations="100" impratio="2"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body pos="0 0 .09"><freejoint/>
        <geom name="ball" type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody><contact><pair geom1="floor" geom2="ball"
      friction="1 .7 .2 .15 .1" condim="{condim}"/></contact></mujoco>""")
  data = mujoco.MjData(model)
  data.qvel[:] = [.3, -.2, 0., .1, -.2, .5]
  data.qacc[:] = [.2, -.1, -1., .1, .2, -.3]
  mujoco.mj_inverse(model, data)
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
                        qvel=data.qvel[None].astype(np.float32),
                        profile="integrated_euler_v1")
  result = _host(mj_inverse(sim, qacc=data.qacc[None].astype(np.float32)))[0]
  np.testing.assert_allclose(result, data.qfrc_inverse, atol=3e-2, rtol=3e-4)
  cc = sim._coupled_constraints
  mask = _host(cc._workspace["pair_mask"][:cc.batch_size * cc.descriptor.npairs])
  pair_map = cc._workspace["pair_maps"]
  slot_map = cc._workspace["slot_maps"]
  packed_pairs = _host(pair_map.packed_to_logical)[0]
  packed_slots = _host(slot_map.packed_to_logical)[0]
  pair_count = int(_host(pair_map.active_count)[0])
  slot_count = int(_host(slot_map.active_count)[0])
  np.testing.assert_array_equal(packed_pairs[:pair_count],
                                np.flatnonzero(mask.reshape(1, -1)[0]))
  slot_flags = _host(cc._workspace["contact_row_data"][:
      cc.batch_size * cc.descriptor.ncontacts_max * 36]).reshape(
          cc.batch_size, cc.descriptor.ncontacts_max, 36)[:, :, 0]
  np.testing.assert_array_equal(packed_slots[:slot_count],
                                np.flatnonzero(slot_flags[0]))
  assert np.all(np.diff(packed_pairs[:pair_count]) >= 0)
  assert np.all(np.diff(packed_slots[:slot_count]) >= 0)


@pytest.mark.parametrize("qpos,qvel,qacc", [
    (-.49, .1, 1.), (-.49, -.1, -1.), (.49, -.1, -1.), (.49, .1, 1.)
])
def test_inverse_dry_friction_and_limit_rows_keep_both_signs(qpos, qvel, qacc):
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="100"/>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0"
      range="-.5 .5" limited="true" frictionloss=".4"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>""")
  data = mujoco.MjData(model)
  data.qpos[:] = [qpos]
  data.qvel[:] = [qvel]
  data.qacc[:] = [qacc]
  mujoco.mj_inverse(model, data)
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
                        qvel=data.qvel[None].astype(np.float32),
                        profile="integrated_euler_v1")
  result = _host(mj_inverse(sim, qacc=data.qacc[None].astype(np.float32)))[0]
  np.testing.assert_allclose(result, data.qfrc_inverse, atol=2e-3, rtol=2e-3)


def test_inverse_uses_supplied_mocap_pose_without_mutating_owned_state():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody><body name="platform" mocap="true">
      <body pos=".5 0 0"><joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".08" mass="1"/>
      </body>
    </body></worldbody></mujoco>""")
  data = mujoco.MjData(model)
  data.qpos[:] = [.4]
  data.qvel[:] = [.2]
  data.qacc[:] = [.7]
  data.mocap_pos[:] = [[.2, -.1, .3]]
  data.mocap_quat[:] = [[1., 0., 0., 0.]]
  mujoco.mj_inverse(model, data)
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  before_qpos = _host(sim.state.qpos)
  before_mocap = _host(sim.state.mocap_pos)
  result = _host(mj_inverse(
      sim, qpos=data.qpos[None].astype(np.float32),
      qvel=data.qvel[None].astype(np.float32),
      qacc=data.qacc[None].astype(np.float32),
      mocap_pos=data.mocap_pos[None].astype(np.float32),
      mocap_quat=data.mocap_quat[None].astype(np.float32)))[0]
  np.testing.assert_allclose(result, data.qfrc_inverse, atol=2e-4, rtol=2e-4)
  np.testing.assert_array_equal(_host(sim.state.qpos), before_qpos)
  np.testing.assert_array_equal(_host(sim.state.mocap_pos), before_mocap)


@pytest.mark.parametrize("iterations", [0, 1])
def test_pinned_low_iteration_budget_advances_finite_contact_iterate(iterations):
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep="0.002" gravity="0 0 -9.81" solver="PGS"
      iterations="{iterations}" tolerance="1e-12"/>
    <worldbody><geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .09"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
      </body></worldbody>
  </mujoco>""")
  native = MetalSimulation(model, profile="integrated_euler_v1")
  native.step(1)
  assert int(_host(native.state.status)[0]) == 0
  assert np.isfinite(_host(native.state.qpos)).all()
  assert np.isfinite(_host(native.state.qvel)).all()
  assert native._coupled_constraints._workspace["out_diagnostics"][1].item() <= iterations

  # MuJoCo advances the finite iterate even when it has not converged within
  # the configured outer budget. Compare the complete one-step state rather
  # than a convergence/status proxy.
  cpu = mujoco.MjData(model)
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(_host(native.state.qpos)[0], cpu.qpos, atol=3e-3, rtol=3e-3)
  np.testing.assert_allclose(_host(native.state.qvel)[0], cpu.qvel, atol=3e-2, rtol=3e-2)
