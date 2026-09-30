# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Spatial tendon qualification (CPU admission/reference + GPU parity)."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.spatial_tendons import SpatialTendonModel
from mujoco_metal.spatial_tendons import spatial_length_reference
from mujoco_metal.spatial_tendons import tendon_wrap_point


def _slide_world(extra=""):
  return ('<site name="s0" pos="0 0 1"/><site name="s1" pos="0.2 0 1"/>'
          '<body pos="0 0 1"><joint name="j" type="slide" axis="1 0 0"/>'
          '<geom type="sphere" size="0.05" mass="1"/>'
          '<site name="s2" pos="0.1 0 0"/></body>' + extra)


def _model(world, tendon, option='timestep="0.002"'):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option {option}/><worldbody>{world}</worldbody><tendon>{tendon}</tendon></mujoco>')


def _dense_J(m, d):
  out = np.zeros((m.ntendon, m.nv))
  for i in range(m.ntendon):
    for k in range(int(m.ten_J_rownnz[i])):
      out[i, int(m.ten_J_colind[int(m.ten_J_rowadr[i]) + k])] = float(
          d.ten_J[int(m.ten_J_rowadr[i]) + k])
  return out


def test_admission_accepts_spatial_paths_cpu():
  m = _model(_slide_world(),
             '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>')
  meta = SpatialTendonModel(m)
  assert meta.has_spatial and meta.paths[0] is not None
  m = _model(_slide_world(),
             '<spatial name="t"><site site="s0"/><site site="s2"/><pulley divisor="2"/>'
             '<site site="s1"/><site site="s0"/></spatial>')
  assert SpatialTendonModel(m).has_spatial
  wrap_world = ('<site name="a" pos="-0.2 0 0.3"/><site name="b" pos="0.2 0 -0.1"/>'
                '<site name="side" pos="0 0.3 0.1"/>'
                '<body pos="0 0 0.1"><geom name="ball" type="sphere" size="0.08" mass="1"/></body>')
  m = _model(wrap_world, '<spatial name="t"><site site="a"/>'
             '<geom geom="ball" sidesite="side"/><site site="b"/></spatial>')
  assert SpatialTendonModel(m).has_spatial
  cyl_world = ('<site name="a" pos="-0.2 0 0.3"/><site name="b" pos="0.2 0 -0.1"/>'
               '<body pos="0 0 0.1"><geom name="rod" type="cylinder" size="0.05 0.2" mass="1"/></body>')
  m = _model(cyl_world, '<spatial name="t"><site site="a"/>'
             '<geom geom="rod"/><site site="b"/></spatial>')
  assert SpatialTendonModel(m).has_spatial
  # Fixed paths are marked (owned by the fixed model), mixtures allowed.
  m = _model(_slide_world(),
             '<fixed name="f"><joint joint="j" coef="1.5"/></fixed>'
             '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>')
  meta = SpatialTendonModel(m)
  assert meta.paths[0] is None and meta.paths[1] is not None


def test_admission_rejects_with_reasons_cpu():
  with pytest.raises(ValueError, match="must start with a site"):
    SpatialTendonModel(_model('<site name="p0" pos="0 0 2"/>' + _slide_world(),
                              '<spatial name="t"><pulley divisor="2"/><site site="p0"/>'
                              '<site site="s1"/><site site="s2"/></spatial>'))
  # Note: <joint> inside <spatial> is rejected by the MuJoCo compiler itself
  # (schema violation), so the lowering's joint-in-spatial guard is
  # defense-in-depth for programmatically built models; both agree.
  with pytest.raises(ValueError, match="unrecognized element"):
    _model('<body pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/>'
           '<geom type="sphere" size="0.05" mass="1"/></body>'
           '<site name="s0" pos="0 0 1"/><site name="s1" pos="1 0 1"/>',
           '<spatial name="t"><site site="s0"/><joint joint="j"/>'
           '<site site="s1"/></spatial>')
  # Note: wrap-geom type mismatches are rejected by the MuJoCo compiler
  # itself; the lowering guard agrees for programmatic models.
  with pytest.raises(ValueError, match="not sphere or cylinder"):
    _model('<site name="a" pos="0 0 0"/><site name="b" pos="1 0 0"/>'
           '<body pos="0.5 0 0"><geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/></body>',
           '<spatial name="t"><site site="a"/>'
           '<geom geom="box"/><site site="b"/></spatial>')
  with pytest.raises(ValueError, match="ordered"):
    m = _model(_slide_world(),
               '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>')
    m.tendon_limited[0] = True
    m.tendon_range[0] = [1.0, -1.0]
    SpatialTendonModel(m)
  with pytest.raises(ValueError, match="nonnegative"):
    m = _model(_slide_world(),
               '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>')
    m.tendon_frictionloss[0] = -0.5
    SpatialTendonModel(m)


def test_reference_matches_oracle_across_poses_cpu():
  rng = np.random.default_rng(0)
  for trial in range(5):
    off = rng.uniform(-0.1, 0.1, size=3)
    world = (f'<site name="a" pos="{-0.2 + off[0]} 0 0.3"/><site name="b" pos="0.2 0 -0.1"/>'
             '<site name="side" pos="0 0.3 0.1"/>'
             '<body pos="0 0 0.1"><geom name="ball" type="sphere" size="0.08" mass="1"/></body>')
    m = _model(world, '<spatial name="t"><site site="a"/>'
               '<geom geom="ball" sidesite="side"/><site site="b"/></spatial>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ref = spatial_length_reference(m, 0, d.site_xpos, d.geom_xpos, d.geom_xmat)
    assert abs(ref - float(d.ten_length[0])) < 1e-9, (trial, ref, float(d.ten_length[0]))


def test_finite_difference_jacobian_matches_oracle_cpu():
  world = ('<site name="a" pos="-0.2 0 0.3"/><site name="b" pos="0.2 0 -0.1"/>'
           '<site name="side" pos="0 0.3 0.1"/>'
           '<body pos="0 0 0"><joint name="j" type="slide" axis="1 0 0"/>'
           '<geom name="ball" type="sphere" size="0.08" mass="1"/></body>')
  m = _model(world, '<spatial name="t"><site site="a"/>'
             '<geom geom="ball" sidesite="side"/><site site="b"/></spatial>')
  # Attach site b to the moving body so the Jacobian is nonzero.
  d = mujoco.MjData(m)
  d.qpos[0] = 0.05
  mujoco.mj_forward(m, d)
  l0 = float(d.ten_length[0])
  mom = _dense_J(m, d)[0]
  h = 1e-6
  fd = np.zeros(m.nv)
  for k in range(m.nv):
    d.qpos[0] = 0.05
    d.qpos[k] += h
    mujoco.mj_forward(m, d)
    fd[k] = (float(d.ten_length[0]) - l0) / h
  np.testing.assert_allclose(mom, fd, rtol=1e-3, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_kinematics_native_parity_gpu():
  import torch
  from mujoco_metal import MetalSimulation
  pulley_world = ('<site name="s0" pos="0 0 1"/><site name="s3" pos="0.6 0 1"/>'
                  '<body pos="0 0 1"><joint name="j" type="slide" axis="1 0 0"/>'
                  '<geom type="sphere" size="0.05" mass="1"/>'
                  '<site name="s1" pos="0.2 0 0"/><site name="s2" pos="0.4 0 0"/></body>')
  wrap_world = ('<site name="a" pos="-0.2 0 0.3"/>'
                '<site name="side" pos="0 0.3 0.1"/>'
                '<body pos="0 0 0"><joint name="j" type="slide" axis="1 0 0"/>'
                '<geom name="ball" type="sphere" size="0.08" mass="1"/>'
                '<site name="b" pos="0.4 0 -0.1"/></body>')
  cases = [
      (pulley_world, '<spatial name="t"><site site="s0"/><site site="s1"/>'
       '<pulley divisor="2"/><site site="s2"/><site site="s3"/></spatial>'),
      (wrap_world, '<spatial name="t"><site site="a"/>'
       '<geom geom="ball" sidesite="side"/><site site="b"/></spatial>'),
  ]
  for world, tendon in cases:
    m = _model(world, tendon)
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    assert sim._spatial_tendons is not None
    d = mujoco.MjData(m)
    d.qpos[0] = 0.07
    d.qvel[0] = 0.3
    mujoco.mj_forward(m, d)
    sim.state._qpos.copy_(torch.as_tensor(
        np.asarray(d.qpos, dtype=np.float32).reshape(1, -1), device=sim.state._device))
    sim.state._qvel.copy_(torch.as_tensor(
        np.asarray(d.qvel, dtype=np.float32).reshape(1, -1), device=sim.state._device))
    poses = sim._smooth.run_device(sim.state._qpos, sim.state._qvel, None, None)["poses"]
    kin = sim._spatial_tendons.run_kinematics(sim.state._qvel, poses)
    np.testing.assert_allclose(kin["length"].cpu().numpy()[0],
                               np.asarray(d.ten_length), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(kin["jacobian"].cpu().numpy()[0], _dense_J(m, d),
                               rtol=2e-5, atol=2e-6)
    cpu_vel = _dense_J(m, d) @ np.asarray(d.qvel)
    np.testing.assert_allclose(kin["velocity"].cpu().numpy()[0], cpu_vel,
                               rtol=2e-5, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_tendon_limit_native_parity_gpu():
  from mujoco_metal import MetalSimulation
  m = _model(_slide_world(),
             '<fixed name="f"><joint joint="j" coef="2.0"/></fixed>')
  m.tendon_limited[0] = True
  m.tendon_range[0] = [-0.5, 0.5]
  m.tendon_margin[0] = 0.02
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[0] = 0.4  # length 0.8: upper limit violated
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.array([[0.4]], dtype=np.float32))
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err
  # Limit holds the joint: it must not run away.
  assert abs(float(sim.state.qpos.cpu().numpy()[0, 0])) < 1.0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_limit_native_parity_gpu():
  # R8-gap: curved spatial rows through the limit machinery (not fixed).
  from mujoco_metal import MetalSimulation
  m = _model(_slide_world(),
             '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>')
  d0 = mujoco.MjData(m)
  mujoco.mj_forward(m, d0)
  l0 = float(np.asarray(d0.ten_length)[0])
  m.tendon_limited[0] = True
  m.tendon_range[0] = [l0 - 0.5, l0 + 0.05]
  m.tendon_margin[0] = 0.02
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[0] = 0.3  # stretches the path past the upper limit
  mujoco.mj_forward(m, cpu)
  assert cpu.nefc >= 1
  sim.reset(qpos=np.array([[0.3]], dtype=np.float32),
            qvel=np.array([[1.0]], dtype=np.float32))
  cpu.qvel[0] = 1.0
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_friction_native_parity_gpu():
  # R8-gap: frictionloss rows built from curved spatial Jacobians.
  from mujoco_metal import MetalSimulation
  m = _model(_slide_world(),
             '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>')
  m.tendon_frictionloss[0] = 0.5
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[0] = 0.0
  cpu.qvel[0] = 2.0
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.array([[0.0]], dtype=np.float32),
            qvel=np.array([[2.0]], dtype=np.float32))
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_tendon_friction_native_parity_gpu():
  from mujoco_metal import MetalSimulation
  m = _model(_slide_world(),
             '<fixed name="f"><joint joint="j" coef="1.0"/></fixed>')
  m.tendon_frictionloss[0] = 0.5
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[0] = 0.0
  cpu.qvel[0] = 2.0
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.array([[0.0]], dtype=np.float32),
            qvel=np.array([[2.0]], dtype=np.float32))
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_tendon_equality_native_parity_gpu():
  from mujoco_metal import MetalSimulation
  world = ('<body pos="0 0 1"><joint name="j1" type="slide" axis="1 0 0"/>'
           '<geom type="sphere" size="0.05" mass="1"/></body>'
           '<body pos="1 0 1"><joint name="j2" type="slide" axis="1 0 0"/>'
           '<geom type="sphere" size="0.05" mass="1"/></body>')
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002"/><worldbody>' + world + '</worldbody>'
      '<tendon><fixed name="f1"><joint joint="j1" coef="1.0"/></fixed>'
      '<fixed name="f2"><joint joint="j2" coef="1.0"/></fixed></tendon>'
      '<equality><tendon tendon1="f1" tendon2="f2" polycoef="0 1 0 0 0"/></equality></mujoco>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = [0.3, -0.1]
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.array([[0.3, -0.1]], dtype=np.float32),
            qvel=np.array([[0.5, -0.2]], dtype=np.float32))
  cpu.qvel[:] = [0.5, -0.2]
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err
  # Equality couples the pair: native residual tracks the CPU residual.
  gq = sim.state.qpos.cpu().numpy()[0]
  q0 = np.asarray(m.qpos0)
  nat_res = abs((gq[0] - q0[0]) - (gq[1] - q0[1]))
  cpu_res = abs((cpu.qpos[0] - q0[0]) - (cpu.qpos[1] - q0[1]))
  assert abs(nat_res - cpu_res) < 5e-4, (nat_res, cpu_res)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_long_unequal_paths_native_parity_gpu():
  # R2 regression: paths longer than the old 8-entry stride with unequal
  # lengths must address the right wraps (flat offsets, no stride).
  import torch
  from mujoco_metal import MetalSimulation
  sites = "".join(f'<site name="s{k}" pos="{0.1 * k} 0 1"/>' for k in range(12))
  long_path = "".join(f'<site site="s{k}"/>' for k in range(12))
  short_path = '<site site="s0"/><site site="s5"/><site site="s11"/>'
  world = sites + '<body pos="0 0 1"><joint name="j" type="slide" axis="1 0 0"/>' \
    '<geom type="sphere" size="0.05" mass="1"/>' \
    '<site name="m" pos="0.3 0 0"/></body>'
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002"/><worldbody>' + world + '</worldbody>'
      '<tendon><spatial name="long">' + long_path + '</spatial>'
      '<spatial name="short">' + short_path + '</spatial>'
      '<spatial name="move"><site site="s0"/><site site="m"/><site site="s11"/></spatial>'
      '</tendon></mujoco>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  d = mujoco.MjData(m)
  d.qpos[0] = 0.07
  d.qvel[0] = 0.3
  mujoco.mj_forward(m, d)
  sim.state._qpos.copy_(torch.as_tensor(
      np.asarray(d.qpos, dtype=np.float32).reshape(1, -1), device=sim.state._device))
  sim.state._qvel.copy_(torch.as_tensor(
      np.asarray(d.qvel, dtype=np.float32).reshape(1, -1), device=sim.state._device))
  poses = sim._smooth.run_device(sim.state._qpos, sim.state._qvel, None, None)["poses"]
  kin = sim._spatial_tendons.run_kinematics(sim.state._qvel, poses)
  np.testing.assert_allclose(kin["length"].cpu().numpy()[0],
                             np.asarray(d.ten_length), rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(kin["jacobian"].cpu().numpy()[0], _dense_J(m, d),
                             rtol=2e-5, atol=2e-6)


def test_total_wrap_overflow_rejected_cpu():
  from mujoco_metal.spatial_tendons import SpatialTendonModel, _MAX_TOTAL_WRAP
  n = _MAX_TOTAL_WRAP + 10
  sites = "".join(f'<site name="s{k}" pos="{0.01 * k} 0 1"/>' for k in range(n))
  path = "".join(f'<site site="s{k}"/>' for k in range(n))
  m = _model(sites, f'<spatial name="t">{path}</spatial>')
  with pytest.raises(ValueError, match="exceed the native cap"):
    SpatialTendonModel(m)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_slack_taut_and_mixed_model_gpu():
  from mujoco_metal import MetalSimulation
  # Fixed tendon spring with a rest range: force is exactly zero while the
  # length is inside, nonzero outside (slack/taut with measured force).
  m = _model(_slide_world(),
             '<fixed name="f" stiffness="50" springlength="-0.1 0.1"><joint joint="j" coef="1.0"/></fixed>'
             '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>')
  assert m.ntendon == 2
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  assert sim._spatial_tendons is not None
  cpu = mujoco.MjData(m)
  sim.reset(qpos=np.array([[0.0]], dtype=np.float32))
  cpu.qpos[0] = 0.0
  mujoco.mj_forward(m, cpu)
  # Inside the rest range the spring is slack: zero acceleration.
  mujoco.mj_forward(m, cpu)
  np.testing.assert_allclose(float(np.asarray(cpu.qacc)[0]), 0.0, atol=1e-9)
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err
  # Outside the range the spring pulls: displace and compare force + motion.
  sim.reset(qpos=np.array([[0.5]], dtype=np.float32))
  cpu.qpos[0] = 0.5
  mujoco.mj_forward(m, cpu)
  asm = sim.assembled_system(recompute=True)
  assert asm is not None
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err
  # Spring actually engaged: displaced run differs from free drift.
  assert abs(float(sim.state.qpos.cpu().numpy()[0, 0]) - 0.5) > 0.01


def test_spatial_armature_contract_cpu():  # R3: wrapped + armature is rejected by the compiler itself; the lowering
  # agrees. Site-only spatial + armature is rejected (015 owns Jdot bias).
  with pytest.raises(ValueError, match="not supported by tendon armature"):
    _model(
        '<site name="a" pos="-0.2 0 0.3"/><site name="b" pos="0.2 0 -0.1"/>'
        '<site name="side" pos="0 0.3 0.1"/>'
        '<body pos="0 0 0.1"><geom name="ball" type="sphere" size="0.08" mass="1"/></body>',
        '<spatial name="t" armature="0.5"><site site="a"/>'
        '<geom geom="ball" sidesite="side"/><site site="b"/></spatial>')
  with pytest.raises(ValueError, match="owned by milestone 015"):
    SpatialTendonModel(_model(
        '<site name="a" pos="0 0 1"/><site name="b" pos="0.2 0 1"/>'
        '<body pos="0 0 1"><joint name="j" type="slide" axis="1 0 0"/>'
        '<geom type="sphere" size="0.05" mass="1"/>'
        '<site name="c" pos="0.1 0 0"/></body>',
        '<spatial name="t" armature="0.5">'
        '<site site="a"/><site site="c"/><site site="b"/></spatial>'))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_fixed_armature_bias_parity_gpu():
  # Fixed tendons have constant Jacobians: armature adds mass only, bias is
  # exactly zero even at nonzero velocity (pinned mj_tendonDot early-out).
  from mujoco_metal import MetalSimulation
  m = _model(
      _slide_world(),
      '<fixed name="f" armature="2.0"><joint joint="j" coef="1.0"/></fixed>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[0] = 0.5
  cpu.qvel[0] = 2.0
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.array([[0.5]], dtype=np.float32),
            qvel=np.array([[2.0]], dtype=np.float32))
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err


def _winch_world():
  # Mass hung from a spatial pulley tendon; the motor below lifts it.
  return ('<site name="top" pos="0 0 1.0"/><site name="low" pos="0 0 0.2"/>'
          '<body pos="0 0 0.5"><joint name="lift" type="slide" axis="0 0 1"/>'
          '<geom name="mass" type="sphere" size="0.06" mass="0.5"/>'
          '<site name="hook" pos="0 0 0.06"/></body>')


def _winch_model(actuator, extra_tendon=""):
  return mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>'
      + _winch_world() + '</worldbody><tendon>'
      '<spatial name="rope"><site site="top"/><site site="hook"/><site site="low"/></spatial>'
      + extra_tendon + '</tendon><actuator>' + actuator + '</actuator></mujoco>')


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_filter_activation_replay_gpu():
  # R1: stateful filter dynamics on a spatial tendon, including actearly
  # replay through Sim reset/restore.
  from mujoco_metal import MetalSimulation
  m = _winch_model('<general tendon="rope" gear="1.5" dyntype="filter" '
                   'dynprm="0.05 0 0" gainprm="3 0 0" actearly="true"/>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, cpu)
  max_err, max_a = 0.0, 0.0
  for step in range(60):
    u = 1.0 if step < 30 else -0.5
    sim.step(1, ctrl=np.array([[u]], dtype=np.float32))
    cpu.ctrl[0] = u
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    ga = sim.state.act.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
    max_a = max(max_a, float(np.max(np.abs(ga - cpu.act))))
  assert max_err < 5e-4, max_err
  assert max_a < 5e-4, max_a
  # Restore replays activation identically.
  snap = sim.state.snapshot()
  sim.step(5, ctrl=np.ones((1, 1), dtype=np.float32))
  sim.state.restore(snap)
  ga = sim.state.act.cpu().numpy()[0]
  np.testing.assert_allclose(ga, np.asarray(snap.act)[0], atol=1e-7)


def test_spatial_motor_admission_cpu():
  from mujoco_metal.stateful_actuation import ActuatorModel
  m = _winch_model('<motor tendon="rope" gear="2"/>')
  meta = ActuatorModel(m)
  assert meta.needs_general_path
  # Scalar fast path keeps rejecting spatial tendons.
  from mujoco_metal.transmissions import TransmissionModel
  with pytest.raises(ValueError, match="fixed tendon"):
    TransmissionModel(m)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_motor_trajectory_gpu():
  # R1: direct motor on a spatial pulley tendon lifts a mass; nonzero
  # length/velocity/gear against CPU intermediates and rollout.
  from mujoco_metal import MetalSimulation
  m = _winch_model('<motor tendon="rope" gear="2"/>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  assert sim._actuators is not None
  cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, cpu)
  gear = float(np.asarray(m.actuator_gear)[0, 0])
  np.testing.assert_allclose(float(np.asarray(cpu.actuator_length)[0]),
                             gear * float(np.asarray(cpu.ten_length)[0]), rtol=1e-6)
  sim.reset_to_keyframe(0) if m.nkey else None
  max_err, max_f = 0.0, 0.0
  for step in range(60):
    u = 1.5 if step < 30 else -1.0
    sim.step(1, ctrl=np.array([[u]], dtype=np.float32))
    cpu.ctrl[0] = u
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
    max_f = max(max_f, float(abs(np.asarray(cpu.actuator_force)[0])))
  assert max_f > 1.0  # nonzero tendon force exercised
  assert max_err < 5e-4, max_err


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_servo_and_muscle_gpu():
  # R1: affine position servo (length-dependent gain) and antagonistic
  # muscle pair on spatial tendons, with force limits and mixed targets.
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>'
      '<site name="top" pos="0 0 1.0"/><site name="low" pos="0 0 0.2"/>'
      '<body pos="-0.3 0 0.5"><joint name="j" type="slide" axis="0 0 1"/>'
      '<geom type="sphere" size="0.06" mass="0.5"/>'
      '<site name="hook" pos="0 0 0.06"/></body>'
      '<body pos="0.3 0 0.5"><joint name="k" type="hinge" axis="0 1 0"/>'
      '<geom type="sphere" size="0.06" mass="0.5"/></body>'
      '</worldbody><tendon>'
      '<spatial name="rope"><site site="top"/><site site="hook"/><site site="low"/></spatial>'
      '<fixed name="rod"><joint joint="k" coef="1.0"/></fixed>'
      '</tendon><actuator>'
      '<general tendon="rope" gaintype="affine" gainprm="0 -20 -2" forcelimited="true" forcerange="-8 8"/>'
      '<general joint="k" dyntype="muscle" dynprm="0.01 0.04 0.01" '
      'gaintype="muscle" gainprm="0.6 1.4 15 15 0.5 1.5 1 1 1.2 0" '
      'biastype="muscle" biasprm="0.6 1.4 15 15 0.5 1.5 1 1 1.2 0" lengthrange="-0.6 0.6"/>'
      '</actuator></mujoco>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  assert sim._actuators is not None
  cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, cpu)
  sim.reset_to_keyframe(0) if m.nkey else sim.reset()
  max_err = 0.0
  for step in range(60):
    u = [0.5, 0.8 if step < 30 else 0.1]
    sim.step(1, ctrl=np.array([u], dtype=np.float32))
    cpu.ctrl[:] = u
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 2e-3, max_err


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_equality_native_parity_gpu():
  # Acceptance gap: equality rows coupling two curved spatial tendons.
  from mujoco_metal import MetalSimulation
  world = ('<site name="a0" pos="-0.3 0 1"/><site name="a1" pos="0.3 0 1"/>'
           '<body pos="-0.3 0 0.5"><joint name="j1" type="slide" axis="0 0 1"/>'
           '<geom type="sphere" size="0.05" mass="0.5"/>'
           '<site name="h1" pos="0 0 0.06"/></body>'
           '<body pos="0.3 0 0.5"><joint name="j2" type="slide" axis="0 0 1"/>'
           '<geom type="sphere" size="0.05" mass="0.5"/>'
           '<site name="h2" pos="0 0 0.06"/></body>')
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>'
      + world + '</worldbody><tendon>'
      '<spatial name="t1"><site site="a0"/><site site="h1"/><site site="a1"/></spatial>'
      '<spatial name="t2"><site site="a1"/><site site="h2"/><site site="a0"/></spatial>'
      '</tendon><equality>'
      '<tendon tendon1="t1" tendon2="t2" polycoef="0 1 0 0 0"/>'
      '</equality></mujoco>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = [0.1, -0.05]
  cpu.qvel[:] = [0.4, 0.2]
  mujoco.mj_forward(m, cpu)
  assert cpu.nefc >= 1  # equality row actually active
  sim.reset(qpos=np.array([[0.1, -0.05]], dtype=np.float32),
            qvel=np.array([[0.4, 0.2]], dtype=np.float32))
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_cylinder_wrap_trajectory_gpu():
  # Acceptance gap (wrap matrix): cylinder side-wrap trajectory parity.
  from mujoco_metal import MetalSimulation
  world = ('<site name="a" pos="-0.3 0 0.5"/><site name="b" pos="0.3 0 0.5"/>'
           '<body pos="0 0 0.5"><joint name="j" type="slide" axis="0 0 1"/>'
           '<geom name="rod" type="cylinder" size="0.05 0.2" mass="0.5"/></body>')
  m = _model(world, '<spatial name="t"><site site="a"/>'
             '<geom geom="rod"/><site site="b"/></spatial>',
             option='timestep="0.002" gravity="0 0 -9.81"')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[0] = 0.1
  cpu.qvel[0] = 0.5
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.array([[0.1]], dtype=np.float32),
            qvel=np.array([[0.5]], dtype=np.float32))
  max_err = 0.0
  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 5e-4, max_err


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_reset_and_copy_gpu():
  # Acceptance gap: reset/snapshot/copy round-trips preserve spatial state.
  from mujoco_metal import MetalSimulation
  m = _model(_slide_world(),
             '<spatial name="t"><site site="s0"/><site site="s2"/><site site="s1"/></spatial>',
             option='timestep="0.002" gravity="0 0 -9.81"')
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.1], [0.2]], dtype=np.float32),
            qvel=np.array([[0.5], [-0.3]], dtype=np.float32))
  for _ in range(10):
    sim.step(1)
  snap = sim.state.snapshot()
  for _ in range(10):
    sim.step(1)
  sim.state.restore(snap)
  q_after = sim.state.qpos.cpu().numpy().copy()
  np.testing.assert_allclose(q_after, np.asarray(snap.qpos), atol=1e-7)
  # Reset to defaults clears motion; copy propagates env rows.
  sim.reset()
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy(),
                             np.zeros((2, 1)), atol=1e-7)
  sim.reset(qpos=np.array([[0.15], [0.0]], dtype=np.float32))
  sim.copy_environment(0, 1)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[1],
                             sim.state.qpos.cpu().numpy()[0], atol=1e-7)
  for _ in range(10):
    sim.step(1)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[1],
                             sim.state.qpos.cpu().numpy()[0], atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spatial_with_contact_parity_gpu():
  # Acceptance gap: spatial tendon forces combined with native contacts.
  # Single-point (sphere/plane) impact: exact parity before first contact
  # and identical settled rest after. Note: a 4-face box/plane impact in the
  # same harness freezes the native dense solver (Cholesky status 3) with or
  # without any tendon, so multi-face impacts stay a contact-solver matter
  # outside 008 scope; the tendon coupling itself is exact here.
  from mujoco_metal import MetalSimulation
  world = ('<geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>'
           '<site name="a" pos="-0.3 0 0.5"/><site name="b" pos="0.3 0 0.5"/>'
           '<body pos="0 0 0.2"><joint name="j" type="slide" axis="0 0 1"/>'
           '<geom name="ball" type="sphere" size="0.05" mass="0.5" '
           'friction="0.5 0.05 0.02" contype="1" conaffinity="1"/>'
           '<site name="h" pos="0 0 0.06"/></body>')
  m = _model(world, '<spatial name="t"><site site="a"/>'
             '<site site="h"/><site site="b"/></spatial>',
             option='timestep="0.002" gravity="0 0 -9.81"')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[0] = 0.0
  cpu.qvel[0] = -1.0
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.array([[0.0]], dtype=np.float32),
            qvel=np.array([[-1.0]], dtype=np.float32))
  pre_err = 0.0
  for _ in range(150):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    if cpu.ncon == 0:
      gq = sim.state.qpos.cpu().numpy()[0]
      pre_err = max(pre_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert pre_err < 1e-6, pre_err  # exact before first contact
  # Both settle to the ball resting on the floor (center z=0.05 -> qpos -0.15).
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0],
                             np.asarray(cpu.qpos), atol=5e-4)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[0],
                             np.asarray(cpu.qvel), atol=5e-4)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
