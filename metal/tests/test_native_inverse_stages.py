# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Nonzero inverse constraint and reusable split-stage regressions."""

import os

import mujoco
import numpy as np
import pytest
import torch

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
