# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU source oracles and opt-in public native ``mj_transmission`` coverage."""

import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def _source_dense_moment(model, data):
  """Expand pinned MuJoCo's packed actuator moment rows to dense form."""
  dense = np.zeros((model.nu, model.nv), dtype=np.float64)
  nnz = np.asarray(data.moment_rownnz, dtype=np.int64)
  rowadr = np.asarray(data.moment_rowadr, dtype=np.int64)
  columns = np.asarray(data.moment_colind, dtype=np.int64)
  values = np.asarray(data.actuator_moment, dtype=np.float64)
  for actuator in range(model.nu):
    start = int(rowadr[actuator])
    for offset in range(int(nnz[actuator])):
      dense[actuator, int(columns[start + offset])] = values[start + offset]
  return dense


def _source_transmission(model, qpos):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  # Explicitly call the source routine after the prerequisite FK/contact/tendon
  # stages, matching its upstream API contract.
  mujoco.mj_transmission(model, data)
  return np.asarray(data.actuator_length).copy(), _source_dense_moment(model, data), data


def _perturbed_qpos(model, dtype=np.float64):
  """Perturb in tangent space so ball/free quaternions stay unit length."""
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  if model.nv:
    qvel = np.linspace(-.03, .04, model.nv, dtype=np.float64)
    mujoco.mj_integratePos(model, qpos, qvel, 1.0)
  return np.asarray(qpos, dtype=dtype)


def _scalar_joint_and_fixed_tendon_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody>
      <body pos="0 0 1"><joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body pos="1 0 1"><joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody>
    <tendon><fixed name="fixed"><joint joint="hinge" coef="1.5"/>
      <joint joint="slide" coef="-.75"/></fixed></tendon>
    <actuator><general name="hinge" joint="hinge" gear="2"/>
      <general name="slide" joint="slide" gear="-.5"/>
      <general name="fixed" tendon="fixed" gear="1.25"/></actuator>
  </mujoco>''')


def _joint_family_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody>
      <body name="hinge_body" pos="0 0 1">
        <joint name="h" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
      <body name="ball_body" pos="1 0 1">
        <joint name="b" type="ball"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
      <body name="free_body" pos="2 0 1">
        <freejoint name="f"/><geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <general name="joint" joint="h" gear="1.25"/>
      <general name="jointinparent" jointinparent="h" gear="-.75"/>
      <general name="ball" joint="b" gear=".2 -.3 .5"/>
      <general name="free" joint="f" gear="1 2 3 0 0 1"/>
    </actuator>
  </mujoco>''')


def _site_and_slider_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody>
      <site name="reference" pos="0 0 1"/>
      <site name="slider" pos="0 0 1"/>
      <body pos="0 0 1">
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="capsule" size=".04 .2" mass=".5"/>
        <site name="tool" pos=".2 .1 .05" quat=".9238795 0 .3826834 0"/>
      </body>
    </worldbody>
    <actuator>
      <general name="site" site="tool" gear="1 .2 -.3 .1 0 .2"/>
      <general name="ref" site="tool" refsite="reference"
               gear=".3 -.2 .1 .2 .1 -.1"/>
      <general name="crank" cranksite="tool" slidersite="slider"
               gear="1.5" cranklength=".12"/>
    </actuator>
  </mujoco>''')


def _spatial_tendon_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody>
      <site name="start" pos="-.2 0 .4"/>
      <site name="end" pos=".3 .1 .2"/>
      <site name="end2" pos=".5 -.1 .3"/>
      <site name="side" pos="0 .3 .2"/>
      <body pos="0 0 .2"><joint name="slide" type="slide" axis="1 0 0"/>
        <geom name="wrap" type="sphere" size=".08" mass="1"/>
        <site name="moving" pos=".1 0 .1"/></body>
    </worldbody>
    <tendon><fixed name="fixed"><joint joint="slide" coef="1.2"/></fixed>
      <spatial name="wrapped"><site site="start"/><geom geom="wrap" sidesite="side"/>
        <site site="end"/></spatial>
      <spatial name="pulley"><site site="start"/><site site="moving"/>
        <pulley divisor="2"/><site site="end"/><site site="end2"/></spatial>
    </tendon>
    <actuator><general name="fixed" tendon="fixed" gear="-.7"/>
      <general name="spatial" tendon="wrapped" gear="1.3"/>
      <general name="pulley" tendon="pulley" gear="-.4"/></actuator>
  </mujoco>''')


def _body_adhesion_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="ball" pos="0 0 .09"><freejoint/>
        <geom name="ball_geom" type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
    <actuator><general name="adhesion" body="ball" gear="0 0 1"/></actuator>
  </mujoco>''')


def _zero_dof_site_model():
  """Valid built-in transmission with actuators but no generalized DOFs."""
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><site name="fixed" pos="1 2 3"/></worldbody>
    <actuator><general name="site" site="fixed" gear="1 0 0 0 0 0"/></actuator>
  </mujoco>''')


def _scalar_motor_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option><flag contact="disable"/></option>
    <worldbody><body pos="0 0 1"><joint name="h" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor name="m" joint="h" gear="1.75"/></actuator>
  </mujoco>''')


def test_source_oracle_dense_expansion_covers_builtin_transmission_families():
  from mujoco_metal.transmissions import TransmissionModel
  from mujoco_metal.transmissions import transmission_workspace_sizes

  cases = (
      (_scalar_joint_and_fixed_tendon_model(), None),
      (_joint_family_model(), None),
      (_site_and_slider_model(), None),
      (_spatial_tendon_model(), None),
      (_body_adhesion_model(), None),
      (_zero_dof_site_model(), None),
  )
  for case_index, (model, _) in enumerate(cases):
    qpos = _perturbed_qpos(model)
    if case_index == 0:
      sizes = transmission_workspace_sizes(
          TransmissionModel(model, allow_inherited=True),
          batch_size=2)
      assert sizes["transmission.position_dims"] == 6
    length, moment, data = _source_transmission(model, qpos)
    assert length.shape == (model.nu,)
    assert moment.shape == (model.nu, model.nv)
    if model.nu:
      assert np.isfinite(length).all() and np.isfinite(moment).all()
    if case_index == 4:
      assert data.ncon > 0
      # Body adhesion's transmission moment uses the current candidate contact
      # direction; checking only its length would miss a zeroed/wrong normal.
      assert np.linalg.norm(moment) > 0


def test_transmission_query_zero_actuator_cpu_transaction_and_empty_shapes():
  torch = pytest.importorskip("torch")
  from mujoco_metal.transmission_query import mj_transmission

  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  sim = SimpleNamespace(
      model=model, batch_size=2,
      state=SimpleNamespace(_device=torch.device("cpu")),
      limits=SimpleNamespace(memory_budget_bytes=1 << 20))
  result = mj_transmission(sim)
  assert tuple(result["actuator_length"].shape) == (2, 0)
  assert tuple(result["actuator_moment"].shape) == (2, 0, 0)
  assert result["status"].tolist() == [0, 0]


def test_transmission_query_preflight_and_finalizer_cpu_edges(monkeypatch):
  torch = pytest.importorskip("torch")
  from mujoco_metal.transmission_query import (
      _checked_counts, _finish_outputs, _preflight,
      _scalar_motor_static_map)

  with pytest.raises(ValueError, match="int32"):
    _checked_counts(1 << 30, 3, 3)
  model = _scalar_joint_and_fixed_tendon_model()
  sim = SimpleNamespace(model=model, batch_size=2,
                        limits=SimpleNamespace(memory_budget_bytes=1))
  with pytest.raises(ValueError, match="memory budget"):
    _preflight(sim, 2, model.nu, model.nv, None)
  sim.limits.memory_budget_bytes = 150
  monkeypatch.setattr("mujoco_metal.finite_difference.estimate_transaction_bytes",
                      lambda _: 32)
  with pytest.raises(ValueError, match="memory budget"):
    _preflight(sim, 2, model.nu, model.nv, None)
  monkeypatch.setattr("mujoco_metal.finite_difference.estimate_transaction_bytes",
                      lambda _: (1 << 31) * 4)
  with pytest.raises(ValueError, match="int32 byte addressing"):
    _preflight(sim, 2, model.nu, model.nv, None)

  motor_model = _scalar_motor_model()
  from mujoco_metal.actuation import ScalarMotorModel
  scalar = ScalarMotorModel.from_model(motor_model)
  qadr, gear, dense = _scalar_motor_static_map(motor_model, scalar)
  np.testing.assert_array_equal(qadr, motor_model.jnt_qposadr[[0]])
  np.testing.assert_array_equal(gear, [1.75])
  np.testing.assert_array_equal(dense, [[1.75]])
  malformed = SimpleNamespace(
      nu=1, nv=1, njnt=0, nq=1,
      actuator_trnid=np.array([[0, -1]], dtype=np.int32),
      jnt_qposadr=np.array([0], dtype=np.int32))
  with pytest.raises(RuntimeError, match="joint map"):
    _scalar_motor_static_map(malformed, scalar)

  monkeypatch.setattr("mujoco_metal.finite_difference.estimate_transaction_bytes",
                      lambda _: 32)
  sim.limits.memory_budget_bytes = 100
  with pytest.raises(ValueError, match="memory budget"):
    _preflight(sim, 2, 1, 1, None, scalar_motor=SimpleNamespace(
        _position_length=None))

  length = torch.zeros((2, 1), dtype=torch.float32)
  moment = torch.ones((2, 1, 2), dtype=torch.float32)
  result = _finish_outputs(length, moment,
                           torch.tensor([0, 0], dtype=torch.int32))
  assert result["status"].tolist() == [0, 0]
  assert result["actuator_moment"] is not moment
  bad = moment.clone()
  bad[1, 0, 0] = float("nan")
  failed = _finish_outputs(length, bad,
                           torch.tensor([0, 0], dtype=torch.int32))
  assert failed["status"].tolist() == [0, 3]
  assert torch.count_nonzero(failed["actuator_moment"][1]).item() == 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
@pytest.mark.parametrize("builder", [
    _scalar_joint_and_fixed_tendon_model,
    _joint_family_model,
    _site_and_slider_model,
    _spatial_tendon_model,
    _body_adhesion_model,
    _zero_dof_site_model,
])
def test_public_native_transmission_matches_pinned_source(builder, monkeypatch):
  import torch
  from mujoco_metal.native_api import mj_transmission
  from mujoco_metal.simulation import MetalSimulation

  model = builder()
  qpos = _perturbed_qpos(model, np.float32)
  batch_qpos = np.stack((qpos, np.asarray(model.qpos0, dtype=np.float32)))
  expected = [_source_transmission(model, row) for row in batch_qpos]
  if builder is _body_adhesion_model:
    assert all(data.ncon > 0 for _, _, data in expected)
  sim = MetalSimulation(model, batch_size=2, qpos=batch_qpos,
                        profile="integrated_euler_v1")
  # Construction may compile model metadata on CPU. Query execution after
  # construction must remain entirely on the prepared native POS path.
  def no_cpu_physics(*args, **kwargs):
    raise AssertionError("transmission query called a CPU physics binding")

  for name in ("mj_forward", "mj_step", "mj_transmission"):
    monkeypatch.setattr(mujoco, name, no_cpu_physics)
  # A nonempty prepared record exercises the query transaction's exact
  # record-identity preservation instead of only checking a None cache.
  prepared = sim.prepare_forward_position(skipsensor=True)
  assert prepared is sim._forward_stages._record
  state_record = sim._forward_stages._record
  state_generation = sim.state.generation
  result = mj_transmission(sim)
  assert result["status"].cpu().tolist() == [0, 0]
  assert result["actuator_length"].shape == (2, model.nu)
  assert result["actuator_moment"].shape == (2, model.nu, model.nv)
  assert sim._forward_stages._record is state_record
  assert sim.state.generation == state_generation
  owned_length = result["actuator_length"].clone()
  owned_moment = result["actuator_moment"].clone()

  for world, (expected_length, expected_moment, _) in enumerate(expected):
    np.testing.assert_allclose(result["actuator_length"][world].cpu().numpy(),
                               expected_length, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(result["actuator_moment"][world].cpu().numpy(),
                               expected_moment, rtol=3e-4, atol=3e-5)

  # Reset invalidates the simulation's reusable workspaces, while query
  # outputs remain independently owned and readable.
  sim.reset()
  assert np.array_equal(result["actuator_length"].cpu().numpy(),
                        owned_length.cpu().numpy())
  assert np.array_equal(result["actuator_moment"].cpu().numpy(),
                        owned_moment.cpu().numpy())


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_public_native_transmission_failure_rolls_back_prepared_record(monkeypatch):
  import numpy as np
  from mujoco_metal.native_api import mj_transmission
  from mujoco_metal.simulation import MetalSimulation

  model = _joint_family_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  record = sim.prepare_forward_position(skipsensor=True)
  assert sim._forward_stages._record is record
  generation = sim.state.generation
  class Plugin:
    def __init__(self):
      self.state = np.array([7], dtype=np.int32)

    def device_snapshot_bytes(self):
      return self.state.nbytes

    def device_snapshot(self):
      return self.state.copy()

    def restore_device(self, snapshot):
      self.state[:] = snapshot

  plugin = Plugin()
  sim._native_plugins = (plugin,)
  original = MetalSimulation.prepare_forward_position

  def fail_after_native_work(self, *args, **kwargs):
    original(self, *args, **kwargs)
    plugin.state[:] += 5
    raise RuntimeError("injected transmission query failure")

  monkeypatch.setattr(MetalSimulation, "prepare_forward_position",
                      fail_after_native_work)
  with pytest.raises(RuntimeError, match="injected transmission query failure"):
    mj_transmission(sim)
  assert sim._forward_stages._record is record
  assert sim.state.generation == generation
  np.testing.assert_array_equal(plugin.state, [7])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_public_native_transmission_scalar_motor_profile(monkeypatch):
  from mujoco_metal.native_api import mj_transmission
  from mujoco_metal.simulation import MetalSimulation

  model = _scalar_motor_model()
  qpos = np.asarray(model.qpos0, dtype=np.float32)[None, :].copy()
  qpos[0, 0] = .23
  expected_length, expected_moment, _ = _source_transmission(model, qpos[0])
  sim = MetalSimulation(model, batch_size=1, qpos=qpos,
                        profile="contact_free_motor_euler_v1")
  def no_cpu_physics(*args, **kwargs):
    raise AssertionError("transmission query called a CPU physics binding")

  for name in ("mj_forward", "mj_step", "mj_transmission"):
    monkeypatch.setattr(mujoco, name, no_cpu_physics)
  sim.prepare_forward_position(skipsensor=True)
  record = sim._forward_stages._record
  result = mj_transmission(sim)
  assert result["status"].cpu().tolist() == [0]
  np.testing.assert_allclose(result["actuator_length"][0].cpu().numpy(),
                             expected_length, rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(result["actuator_moment"][0].cpu().numpy(),
                             expected_moment, rtol=1e-5, atol=1e-6)
  assert sim._forward_stages._record is record


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_public_native_transmission_tracks_reset_and_restored_state(monkeypatch):
  import torch
  from mujoco_metal.native_api import mj_transmission
  from mujoco_metal.simulation import MetalSimulation

  model = _scalar_joint_and_fixed_tendon_model()
  qpos0 = np.asarray(model.qpos0, dtype=np.float32)[None, :].copy()
  qpos1 = qpos0.copy()
  qpos0[0, 0], qpos0[0, 1] = .17, -.21
  qpos1[0, 0], qpos1[0, 1] = -.34, .29
  expected0 = _source_transmission(model, qpos0[0])[:2]
  expected1 = _source_transmission(model, qpos1[0])[:2]
  sim = MetalSimulation(model, batch_size=1, qpos=qpos0,
                        profile="integrated_euler_v1")
  checkpoint = sim.snapshot()

  def no_cpu_physics(*args, **kwargs):
    raise AssertionError("transmission query called a CPU physics binding")

  for name in ("mj_forward", "mj_step", "mj_transmission"):
    monkeypatch.setattr(mujoco, name, no_cpu_physics)
  sim.reset(qpos=qpos1)
  after_reset = mj_transmission(sim)
  np.testing.assert_allclose(after_reset["actuator_length"][0].cpu().numpy(),
                             expected1[0], rtol=2e-4, atol=2e-5)
  np.testing.assert_allclose(after_reset["actuator_moment"][0].cpu().numpy(),
                             expected1[1], rtol=3e-4, atol=3e-5)
  sim.restore(checkpoint)
  after_restore = mj_transmission(sim)
  np.testing.assert_allclose(after_restore["actuator_length"][0].cpu().numpy(),
                             expected0[0], rtol=2e-4, atol=2e-5)
  np.testing.assert_allclose(after_restore["actuator_moment"][0].cpu().numpy(),
                             expected0[1], rtol=3e-4, atol=3e-5)
  # Results from both calls remain owned across subsequent state mutations.
  before_reset = after_reset["actuator_length"].clone()
  sim.reset(qpos=qpos0)
  assert torch.equal(after_reset["actuator_length"], before_reset)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_public_native_transmission_isolates_nonfinite_world(monkeypatch):
  from mujoco_metal.native_api import mj_transmission
  from mujoco_metal.simulation import MetalSimulation

  model = _joint_family_model()
  qpos = np.stack((np.asarray(model.qpos0, dtype=np.float32),
                   np.asarray(model.qpos0, dtype=np.float32)))
  expected_length, expected_moment, _ = _source_transmission(model, qpos[1])
  sim = MetalSimulation(model, batch_size=2, qpos=qpos,
                        profile="integrated_euler_v1")

  def no_cpu_physics(*args, **kwargs):
    raise AssertionError("transmission query called a CPU physics binding")

  for name in ("mj_forward", "mj_step", "mj_transmission"):
    monkeypatch.setattr(mujoco, name, no_cpu_physics)
  sim.state._qpos[0, 0] = float("nan")
  result = mj_transmission(sim)
  status = result["status"].cpu().tolist()
  assert status[0] != 0 and status[1] == 0
  assert np.count_nonzero(result["actuator_length"][0].cpu().numpy()) == 0
  assert np.count_nonzero(result["actuator_moment"][0].cpu().numpy()) == 0
  np.testing.assert_allclose(result["actuator_length"][1].cpu().numpy(),
                             expected_length, rtol=2e-4, atol=2e-5)
  np.testing.assert_allclose(result["actuator_moment"][1].cpu().numpy(),
                             expected_moment, rtol=3e-4, atol=3e-5)
