"""Pinned MuJoCo oracle tests for the bundled touch-grid lowering."""

import pytest
import os

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from mujoco_metal.bundled_plugins import lower_bundled_plugins
from mujoco_metal.bundled_touch_grid import (
    lower_touch_grid_sensors,
    touch_grid_reference,
    touch_grid_workspace_sizes,
)


@pytest.fixture(scope="module", autouse=True)
def _load_plugins():
  assert mujoco.__version__ == "3.10.0"
  mujoco.mj_loadAllPluginLibraries(mujoco.PLUGINS_DIR)


def _model(*, cone="elliptic", condim=3, integrator="Euler",
           extra_pad_geom=""):
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.002" gravity="0 0 -9.81" integrator="{integrator}"
              cone="{cone}"/>
      <extension><plugin plugin="mujoco.sensor.touch_grid"/></extension>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="pad" pos="0 0 .08">
          <freejoint/><geom name="pad_geom" type="box" size=".12 .12 .04"
              mass="1" condim="{condim}"/>
          {extra_pad_geom}
          <site name="touch" size=".1 .1 .1"/>
        </body>
      </worldbody>
      <sensor><plugin name="grid" plugin="mujoco.sensor.touch_grid"
          objtype="site" objname="touch">
        <config key="nchannel" value="3"/>
        <config key="size" value="5 4"/>
        <config key="fov" value="150 80"/>
        <config key="gamma" value=".3"/>
      </plugin></sensor>
    </mujoco>
  """)


def _rows_from_pinned(model, data):
  rows = []
  for i in range(int(data.ncon)):
    contact = data.contact[i]
    wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, i, wrench)
    rows.append(np.concatenate((
        [1.0, float(contact.geom1), float(contact.geom2)],
        np.asarray(contact.frame, dtype=np.float64),
        np.asarray(contact.pos, dtype=np.float64), wrench)))
  return np.asarray(rows, dtype=np.float64).reshape(-1, 21)


def test_touch_grid_lowering_and_contact_histogram_match_pinned_callback():
  model = _model()
  bundled = lower_bundled_plugins(model)
  sensors = lower_touch_grid_sensors(model, bundled)
  assert len(sensors) == 1
  sensor = sensors[0]
  assert sensor.dimension == int(model.sensor_dim[0]) == 60

  data = mujoco.MjData(model)
  data.qpos[2] = .03
  mujoco.mj_forward(model, data)
  mujoco.mj_step(model, data)
  assert data.ncon > 0
  quat = np.empty(4, dtype=np.float64)
  mat = np.asarray(data.site_xmat[sensor.site_id]).reshape(3, 3)
  mujoco.mju_mat2Quat(quat, mat.reshape(-1))
  output = touch_grid_reference(
      sensor, contact_rows=_rows_from_pinned(model, data),
      site_pos=data.site_xpos[sensor.site_id], site_quat=quat,
      geom_bodies=model.geom_bodyid, body_weldid=model.body_weldid)
  expected = data.sensordata[sensor.address:sensor.address + sensor.dimension]
  np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
  assert np.any(np.abs(output) > 0)


def test_touch_grid_runtime_preflight_accounts_exact_fixed_backings():
  from mujoco_metal.capacity import estimate_capacity
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model()
  bundled = lower_bundled_plugins(model)
  rows = lower_coupled_constraints(model)
  sizes = touch_grid_workspace_sizes(
      model, 2, rows.ncontacts_max, bundled_plugins=bundled,
      pair_capacity=rows.npairs)
  assert sizes["contact_active"] == 2 * max(rows.ncontacts_max, 1)
  assert sizes["contact_mask_source"] == 2 * max(rows.ncontacts_max, 1)
  assert sizes["empty_contact_force"] == 2 * max(rows.ncontacts_max, 1) * 11
  assert sizes["contact_condim"] == 3 * max(rows.ncontacts_max, 1)
  assert sizes["contact_mu"] == 5 * max(rows.ncontacts_max, 1)
  assert sizes["dims"] == 10
  assert sizes["empty_contact_pair_geoms"] == 2 * max(rows.npairs, 1)

  estimate = estimate_capacity(
      model, 2, rows.npairs, rows.ncontacts_max, rows.nr,
      nr_joint=rows.nr_joint, touch_grid_contact_capacity=rows.ncontacts_max,
      touch_grid_pair_capacity=rows.npairs)
  breakdown = dict(estimate.memory_breakdown)
  assert breakdown["sensor.touch_grid.empty_contact_force"] == (
      4 * sizes["empty_contact_force"])
  assert breakdown["sensor.touch_grid.contact_mu"] == 4 * sizes["contact_mu"]
  assert breakdown["sensor.touch_grid.empty_contact_pair_geoms"] == (
      4 * sizes["empty_contact_pair_geoms"])
  # Public admission must classify the bundled sensor as an ACC producer;
  # construction then allocates the corresponding native stage consumer.
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert "touch_grid ACC sensors" in " ".join(profile.supported)


def test_touch_grid_capacity_uses_compiled_candidate_pairs_not_model_npair():
  from mujoco_metal.capacity import estimate_capacity
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  model = _model(extra_pad_geom=(
      '<geom name="pad_sphere" type="sphere" size=".035" '
      'pos=".03 0 .05"/>'))
  rows = lower_coupled_constraints(model)
  assert int(model.npair) == 0
  assert int(rows.npairs) == 2
  sizes = touch_grid_workspace_sizes(
      model, 3, rows.ncontacts_max,
      bundled_plugins=lower_bundled_plugins(model),
      pair_capacity=rows.npairs)
  assert sizes["empty_contact_pair_geoms"] == 2 * rows.npairs

  estimate = estimate_capacity(
      model, 3, rows.npairs, rows.ncontacts_max, rows.nr,
      nr_joint=rows.nr_joint,
      touch_grid_contact_capacity=rows.ncontacts_max,
      touch_grid_pair_capacity=rows.npairs)
  breakdown = dict(estimate.memory_breakdown)
  assert breakdown["sensor.touch_grid.empty_contact_pair_geoms"] == (
      4 * 2 * rows.npairs)

  expanded = estimate_capacity(
      model, 3, rows.npairs + 2, rows.ncontacts_max, rows.nr,
      nr_joint=rows.nr_joint,
      touch_grid_contact_capacity=rows.ncontacts_max,
      touch_grid_pair_capacity=rows.npairs + 2)
  expanded_breakdown = dict(expanded.memory_breakdown)
  assert (expanded_breakdown["sensor.touch_grid.empty_contact_pair_geoms"]
          - breakdown["sensor.touch_grid.empty_contact_pair_geoms"]
          == 4 * 2 * 2)


def test_touch_grid_contact_mask_is_staged_to_contiguous_owned_storage():
  torch = pytest.importorskip("torch")
  from types import SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation

  batch, capacity = 2, 5
  source_storage = torch.arange(batch * capacity * 2, dtype=torch.float32)
  # This mirrors CC's activity lane view: correct values, but row-strided.
  source = source_storage.reshape(batch, capacity, 2)[:, :, 0]
  assert not source.is_contiguous()
  target = torch.zeros((batch, capacity), dtype=torch.float32)
  sim = SimpleNamespace(
      _contact_views=lambda: {"contact_mask": source},
      _touch_grid=SimpleNamespace(_contact_mask_source=target))

  staged = MetalSimulation._touch_grid_contact_views(sim)
  assert staged["contact_mask"] is target
  assert target.is_contiguous()
  torch.testing.assert_close(target, source)
  assert not torch.equal(target, source_storage.reshape(batch, capacity, 2)[:, :, 1])


def test_inverse_query_transaction_restores_bundled_program_roots():
  torch = pytest.importorskip("torch")
  from types import SimpleNamespace
  from mujoco_metal.native_api import _inverse_query_workspaces

  touch = SimpleNamespace(
      _status=torch.tensor([0, 3], dtype=torch.int32),
      _contact_active=torch.tensor([[1, 0], [0, 1]], dtype=torch.int32),
      _contact_mask_source=torch.tensor([[1., 0.], [0., 1.]]),
      _contact_frame=torch.arange(48, dtype=torch.float32).reshape(2, 2, 12),
      _contact_force=torch.arange(44, dtype=torch.float32).reshape(2, 2, 11),
      _contact_condim=torch.ones((2, 3), dtype=torch.int32),
      _contact_mu=torch.ones((2, 5), dtype=torch.float32),
      _slot_pair=torch.tensor([0, -1], dtype=torch.int32))
  cable = SimpleNamespace(_flags=torch.tensor([1], dtype=torch.int32))
  history = SimpleNamespace(_history=torch.arange(6, dtype=torch.int64))
  delay = SimpleNamespace(_buffer=torch.arange(8, dtype=torch.float32))
  control = torch.tensor([[.2], [.7]], dtype=torch.float32)
  act_velocity = torch.tensor([[.1, -.2], [.3, .4]], dtype=torch.float32)
  act_force = torch.tensor([[1., -2.], [3., -4.]], dtype=torch.float32)
  act_qfrc = torch.tensor([[5., -6., 7.], [-8., 9., -10.]], dtype=torch.float32)
  sim = SimpleNamespace(
      _touch_grid=touch, _cable=cable, _history_program=history,
      _delay=delay, _control=control, _native_plugins=(),
      _act_vel=act_velocity, _sen_act_force=act_force,
      _sen_qfrc_act=act_qfrc)
  tensors = {
      "touch_status": touch._status,
      "touch_mask": touch._contact_mask_source,
      "touch_frame": touch._contact_frame,
      "touch_force": touch._contact_force,
      "touch_condim": touch._contact_condim,
      "touch_mu": touch._contact_mu,
      "touch_slots": touch._slot_pair,
      "cable_flags": cable._flags,
      "history": history._history,
      "delay": delay._buffer,
      "control": control,
      "act_velocity": act_velocity,
      "act_force": act_force,
      "act_qfrc": act_qfrc,
  }
  saved = {name: tensor.clone() for name, tensor in tensors.items()}
  with _inverse_query_workspaces(sim):
    for tensor in tensors.values():
      tensor.zero_()
  for name, tensor in tensors.items():
    assert tensor is {
        "touch_status": touch._status,
        "touch_mask": touch._contact_mask_source,
        "touch_frame": touch._contact_frame,
        "touch_force": touch._contact_force,
        "touch_condim": touch._contact_condim,
        "touch_mu": touch._contact_mu,
        "touch_slots": touch._slot_pair,
        "cable_flags": cable._flags,
        "history": history._history,
        "delay": delay._buffer,
        "control": control,
        "act_velocity": act_velocity,
        "act_force": act_force,
        "act_qfrc": act_qfrc,
    }[name]
    torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)


@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_touch_grid_fixture_compiles_all_supported_contact_rows(cone, condim):
  model = _model(cone=cone, condim=condim)
  sensor = lower_touch_grid_sensors(
      model, lower_bundled_plugins(model))[0]
  assert sensor.dimension == int(model.sensor_dim[0])
  assert int(model.opt.cone) == int(
      getattr(mujoco.mjtCone, f"mjCONE_{cone.upper()}"))
  assert int(model.geom_condim[1]) == condim


@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"),
    ("RK4", "integrated_rk4_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
])
def test_touch_grid_public_admission_covers_all_integrators(integrator, profile):
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(integrator=integrator)
  decision = validate_stepping_profile(model, profile=profile)
  assert decision.name == profile
  assert "touch_grid ACC sensors" in " ".join(decision.supported)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native touch_grid composition")
@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_native_integrated_touch_grid_sensor_matches_pinned_step_samples(
    cone, condim):
  from mujoco_metal import MetalSimulation

  model = _model(cone=cone, condim=condim)
  references = [mujoco.MjData(model), mujoco.MjData(model)]
  references[0].qpos[2] = .03
  references[1].qpos[2] = .35  # healthy inactive-contact peer
  simulation = MetalSimulation(
      model, batch_size=2,
      qpos=np.stack([data.qpos for data in references]).astype(np.float32),
      qvel=np.stack([data.qvel for data in references]).astype(np.float32),
      profile="integrated_euler_v1")
  sensor = lower_touch_grid_sensors(
      model, lower_bundled_plugins(model))[0]
  observed_nonzero = False
  for step in range(4):
    for reference in references:
      mujoco.mj_step(model, reference)
    status = simulation.step()
    assert np.all(status.detach().cpu().numpy() == 0), f"status at step {step}"
    actual_all = simulation.step_sensordata()
    for world, reference in enumerate(references):
      actual = actual_all[world,
          sensor.address:sensor.address + sensor.dimension]
      expected = reference.sensordata[
          sensor.address:sensor.address + sensor.dimension]
      np.testing.assert_allclose(actual, expected,
                                 rtol=2e-4, atol=2e-5,
                                 err_msg=f"{cone}/{condim} world {world} step {step}")
      if world == 0:
        observed_nonzero |= bool(np.any(np.abs(actual) > 0))
      np.testing.assert_allclose(
          simulation.state.qpos[world].detach().cpu().numpy(), reference.qpos,
          rtol=3e-4, atol=2e-5)
      np.testing.assert_allclose(
          simulation.state.qvel[world].detach().cpu().numpy(), reference.qvel,
          rtol=3e-4, atol=2e-5)
  assert observed_nonzero, "native touch_grid path never published a contact load"

  checkpoint = simulation.snapshot()
  for reference in references:
    mujoco.mj_step(model, reference)
  assert np.all(simulation.step().detach().cpu().numpy() == 0)
  expected_qpos = simulation.state.qpos.detach().cpu().numpy().copy()
  expected_qvel = simulation.state.qvel.detach().cpu().numpy().copy()
  expected_sensor = simulation.step_sensordata().copy()
  simulation.restore(checkpoint)
  for reference in references:
    mujoco.mj_step(model, reference)
  assert np.all(simulation.step().detach().cpu().numpy() == 0)
  np.testing.assert_array_equal(simulation.state.qpos.detach().cpu().numpy(), expected_qpos)
  np.testing.assert_array_equal(simulation.state.qvel.detach().cpu().numpy(), expected_qvel)
  np.testing.assert_array_equal(simulation.step_sensordata(), expected_sensor)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native corrupted touch-grid pair guard")
@pytest.mark.parametrize("bad_pair_kind", ["high", "negative"])
def test_native_touch_grid_rejects_corrupt_contact_pair_without_oob_read(
    bad_pair_kind):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.forward_stages import ForwardStage

  model = _model()
  reference = mujoco.MjData(model)
  reference.qpos[2] = .03
  sim = MetalSimulation(
      model, batch_size=1, qpos=reference.qpos[None].astype(np.float32),
      qvel=reference.qvel[None].astype(np.float32),
      profile="integrated_euler_v1")
  pos_record = sim.prepare_forward_position()
  poses = pos_record.values[ForwardStage.POS]["poses"]
  contact = sim._touch_grid_contact_views()
  assert contact["contact_mask"].is_contiguous()
  contact["contact_mask"].zero_()
  contact["contact_mask"][0, 0] = 1.0
  contact["slot_pair"].fill_(
      sim._touch_grid.pair_capacity if bad_pair_kind == "high" else -2)
  out = sim._state._torch.zeros(
      (1, model.nsensordata), dtype=sim._state._torch.float32,
      device=sim._state._device)
  sim._touch_grid.run_device(contact, poses, out)
  assert int(sim._touch_grid._status[0].detach().cpu().item()) == 3


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native touch_grid integrator matrix")
@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"),
    ("RK4", "integrated_rk4_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
])
def test_native_touch_grid_sensor_composes_with_all_integrators(
    integrator, profile):
  from mujoco_metal import MetalSimulation

  model = _model(integrator=integrator)
  references = [mujoco.MjData(model), mujoco.MjData(model)]
  references[0].qpos[2] = .03
  references[1].qpos[2] = .35
  sim = MetalSimulation(
      model, batch_size=2,
      qpos=np.stack([data.qpos for data in references]).astype(np.float32),
      qvel=np.stack([data.qvel for data in references]).astype(np.float32),
      profile=profile)
  sensor = lower_touch_grid_sensors(
      model, lower_bundled_plugins(model))[0]
  saw_contact = False
  for step in range(4):
    for reference in references:
      mujoco.mj_step(model, reference)
    status = sim.step()
    assert np.all(status.detach().cpu().numpy() == 0), (integrator, step)
    data = sim.step_sensordata()
    for world, reference in enumerate(references):
      actual = data[world, sensor.address:sensor.address + sensor.dimension]
      np.testing.assert_allclose(
          actual,
          reference.sensordata[sensor.address:sensor.address + sensor.dimension],
          rtol=3e-4, atol=3e-5,
          err_msg=f"{integrator} sensor world {world} step {step}")
      np.testing.assert_allclose(
          sim.state.qpos[world].detach().cpu().numpy(), reference.qpos,
          rtol=5e-4, atol=3e-5)
      np.testing.assert_allclose(
          sim.state.qvel[world].detach().cpu().numpy(), reference.qvel,
          rtol=5e-4, atol=5e-5)
      if world == 0:
        saw_contact |= bool(np.any(np.abs(actual) > 0))
  assert saw_contact
