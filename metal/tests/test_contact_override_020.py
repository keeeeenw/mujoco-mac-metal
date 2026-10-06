# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned global contact-override lowering and native regression gates."""
from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.stepping import validate_stepping_profile


_OVERRIDE = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)


def _contact_model(pair_mode, condim, cone, z=0.18):
  pair = (f'<contact><pair geom1="floor" geom2="ball" condim="{condim}" '
          'friction=".91 .73 .61 .47 .39" solref=".031 .82" '
          'solimp=".61 .89 .017 .57 2.3" margin=".006" gap=".004" '
          'solreffriction=".043 .76"/></contact>'
          if pair_mode == "explicit" else "")
  geom_condim = condim if pair_mode == "dynamic" else 3
  xml = f'''<mujoco><option timestep=".002" gravity="0 0 0"
      cone="{cone}" iterations="120" tolerance="1e-8"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" condim="{geom_condim}"
          friction=".12 .08 .04" margin=".001" gap=".003"/>
      <body name="ball" pos=".02 -.01 {z}"><freejoint/>
        <geom name="ball" type="sphere" size=".2" condim="{geom_condim}"
            friction=".32 .21 .11" margin=".002" gap=".005"/>
      </body>
    </worldbody>{pair}</mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  return model


def _set_override(model, *, margin=0.023, friction=None):
  model.opt.enableflags |= _OVERRIDE
  model.opt.o_margin = margin
  model.opt.o_solref[:] = [.013, .79]
  model.opt.o_solimp[:] = [.72, .93, .011, .63, 2.15]
  model.opt.o_friction[:] = (
      [.37, .29, .23, .19, .11] if friction is None else friction)


def _mixed_rigid_flex_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" cone="elliptic"
        iterations="100" tolerance="1e-8"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" contype="0"
          conaffinity="1" condim="4" friction=".8 .5 .3"/>
      <body name="ball" pos="-.25 0 .045"><freejoint/>
        <geom name="ball" type="sphere" size=".05" mass=".5"
            contype="1" conaffinity="0" condim="4"
            friction=".8 .5 .3"/>
      </body>
      <flexcomp name="cloth" type="grid" count="2 2 1"
          pos=".2 0 .025" spacing=".08 .08 .01" mass=".4" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"
            condim="4" friction=".8 .5 .3"/>
        <elasticity young="100" poisson=".2" thickness=".01"/>
      </flexcomp>
    </worldbody>
  </mujoco>''')


def _cpu_float32_rollout(model, qpos, qvel, steps):
  """Run a pinned CPU trajectory from the exact public float32 inputs."""
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(qpos, dtype=np.float32).astype(np.float64)
  data.qvel[:] = np.asarray(qvel, dtype=np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  initial_qfrc_constraint = data.qfrc_constraint.copy()
  for _ in range(steps):
    mujoco.mj_step(model, data)
  return {
      "qpos": data.qpos.copy(),
      "qvel": data.qvel.copy(),
      "qacc": data.qacc.copy(),
      "initial_qfrc_constraint": initial_qfrc_constraint,
  }


@pytest.mark.parametrize("pair_mode", ["explicit", "dynamic"])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_cpu_override_trajectory_uses_same_float32_public_initial_state(
    pair_mode, cone, condim):
  """The pinned CPU oracle sees real trajectory changes from the override."""
  model = _contact_model(pair_mode, condim, cone)
  _set_override(model)
  baseline = _contact_model(pair_mode, condim, cone)
  qpos = np.asarray(model.qpos0, dtype=np.float32).copy()
  qvel = np.zeros(model.nv, dtype=np.float32)
  qvel[:3] = [.3, -.2, -1.1]
  qvel[3:6] = [1.3, -2.1, -.5]

  changed = _cpu_float32_rollout(model, qpos, qvel, 8)
  unchanged = _cpu_float32_rollout(baseline, qpos, qvel, 8)
  assert np.all(np.isfinite(changed["qacc"]))
  assert np.linalg.norm(changed["initial_qfrc_constraint"]) > 1e-5
  delta = max(float(np.max(np.abs(changed[name] - unchanged[name])))
              for name in ("qpos", "qvel", "qacc"))
  assert delta > 1e-7


@pytest.mark.parametrize("pair_mode", ["explicit", "dynamic"])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_global_override_lowering_matches_pinned_assignments(
    pair_mode, cone, condim):
  model = _contact_model(pair_mode, condim, cone)
  _set_override(model)
  desc = lower_coupled_constraints(model)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  assert desc.npairs == 1
  assert desc.condim.tolist() == [condim]
  assert desc.gap.tolist() == pytest.approx(
      [.004] if pair_mode == "explicit" else [.008])
  assert desc.margin.tolist() == pytest.approx([.023])
  np.testing.assert_allclose(desc.solref[0], [.013, .79], rtol=0, atol=3e-8)
  np.testing.assert_allclose(desc.solreffriction[0], [.013, .79], rtol=0, atol=3e-8)
  np.testing.assert_allclose(desc.solimp[0], [.72, .93, .011, .63, 2.15],
                             rtol=0, atol=1.1e-7)
  np.testing.assert_allclose(desc.friction[0], [.37, .29, .23, .19, .11],
                             rtol=0, atol=3e-8)
  np.testing.assert_array_equal(desc.contact_condim,
                                np.full(desc.ncontacts_max, condim))
  np.testing.assert_array_equal(
      desc.contact_friction,
      np.tile(np.asarray([.37, .29, .23, .19, .11], np.float32),
              (desc.ncontacts_max, 1)))
  np.testing.assert_array_equal(
      desc.contact_solreffriction,
      np.tile(np.asarray([.013, .79], np.float32),
              (desc.ncontacts_max, 1)))

  contact = next(c for c in data.contact[:data.ncon]
                 if set(map(int, c.geom)) == {0, 1})
  assert int(contact.dim) == condim
  assert float(contact.includemargin) == pytest.approx(.023)
  np.testing.assert_allclose(contact.solref, [.013, .79], rtol=0, atol=3e-8)
  np.testing.assert_allclose(contact.solreffriction, [.013, .79], rtol=0, atol=3e-8)
  np.testing.assert_allclose(contact.solimp, [.72, .93, .011, .63, 2.15],
                             rtol=0, atol=1.1e-7)
  np.testing.assert_allclose(contact.friction, [.37, .29, .23, .19, .11],
                             rtol=0, atol=3e-8)


def test_override_friction_clamps_each_of_five_coefficients_and_keeps_gap():
  model = _contact_model("explicit", 6, "elliptic")
  _set_override(model, margin=.031, friction=[-1.0, 0.0, .2, 1e-7, .4])
  desc = lower_coupled_constraints(model)
  np.testing.assert_allclose(desc.friction[0], [1e-5, 1e-5, .2, 1e-5, .4],
                             rtol=0, atol=1e-8)
  np.testing.assert_array_equal(desc.contact_friction[0], desc.friction[0])
  assert float(desc.margin[0]) == pytest.approx(.031)
  assert float(desc.gap[0]) == pytest.approx(.004)
  assert int(desc.condim[0]) == 6


def test_override_replaces_margin_for_nonpenetrating_pair_without_replacing_gap():
  model = _contact_model("explicit", 3, "pyramidal", z=.205)
  _set_override(model, margin=.02)
  desc = lower_coupled_constraints(model)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  contact = next(c for c in data.contact[:data.ncon]
                 if set(map(int, c.geom)) == {0, 1})
  assert contact.dist > 0.0
  assert contact.dist < float(contact.includemargin)
  assert float(contact.includemargin) == pytest.approx(.02)
  assert float(desc.margin[0]) == pytest.approx(.02)
  assert float(desc.gap[0]) == pytest.approx(.004)
  assert not bool(contact.exclude)


def test_integrated_profile_admits_override_and_binds_mutation_restore():
  from mujoco_metal.device_state import DeviceState

  model = _contact_model("dynamic", 4, "elliptic")
  _set_override(model)
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert profile.name == "integrated_euler_v1"
  assert any("global contact overrides" in item for item in profile.supported)

  saved = model.opt.o_solref.copy()
  model.opt.o_solref[0] = .017
  changed = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert changed.model_fingerprint != profile.model_fingerprint
  with pytest.raises(ValueError, match="profile does not match the compiled model"):
    DeviceState(model, profile, batch_size=1, device="cpu")
  model.opt.o_solref[:] = saved
  restored = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert restored == profile


@pytest.mark.parametrize("field", [
    "enableflags", "o_margin", "o_solref", "o_solimp", "o_friction",
])
def test_live_override_binding_rejects_before_public_entry_mutation(field):
  """Every supported entry point rejects source-option drift before effects."""
  from types import SimpleNamespace
  from mujoco_metal.simulation import (
      MetalSimulation, _contact_override_option_signature)

  model = _contact_model("dynamic", 3, "elliptic")
  _set_override(model)
  sim = object.__new__(MetalSimulation)
  sim._mjmodel = model
  sim._contact_override_binding = _contact_override_option_signature(model)
  # These sentinels model state and retained caches. Each entry point must
  # fail at its first line, without even inspecting or changing them.
  sim._state = SimpleNamespace(generation=17)
  sim._assembled_system_valid = True
  sim._accepted_step = object()
  sim._last_coupled = object()
  before = (sim._state.generation, sim._assembled_system_valid,
            sim._accepted_step, sim._last_coupled)
  if field == "enableflags":
    model.opt.enableflags &= ~_OVERRIDE
  elif field == "o_margin":
    model.opt.o_margin += .001
  else:
    getattr(model.opt, field)[0] += .001

  calls = (
      lambda: sim.step(1),
      lambda: sim.assembled_system(),
      lambda: sim.snapshot(),
      lambda: sim.restore({}),
  )
  for call in calls:
    with pytest.raises(ValueError, match="changed after compilation"):
      call()
    assert (sim._state.generation, sim._assembled_system_valid,
            sim._accepted_step, sim._last_coupled) == before


@pytest.mark.parametrize("field,value,match", [
    ("o_solref", np.array([np.nan, .7]), "solref must be finite"),
    ("o_solimp", np.array([.7, .9, np.inf, .5, 2.]), "solimp must be finite"),
    ("o_friction", np.array([.3, .2, 1e300, .1, .1]),
     "friction must be float32-representable"),
    ("o_margin", np.array([np.inf]), "margin must be finite"),
])
def test_invalid_override_payload_fails_preflight(field, value, match):
  model = _contact_model("dynamic", 3, "pyramidal")
  _set_override(model)
  if field == "o_margin":
    model.opt.o_margin = float(value[0])
  else:
    getattr(model.opt, field)[:] = value
  with pytest.raises(ValueError, match=match):
    lower_coupled_constraints(model)
  with pytest.raises(ValueError, match=match):
    validate_stepping_profile(model, profile="integrated_euler_v1")


def test_override_is_applied_to_existing_flex_contact_lowering():
  from mujoco_metal.flex_contact import lower_flex_contacts

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 .1"
          spacing=".1 .1 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"/>
      </flexcomp></worldbody></mujoco>''')
  _set_override(model, margin=.029)
  desc = lower_flex_contacts(model)
  assert desc.slot_count > 0
  np.testing.assert_allclose(desc.solref[0], [.013, .79], rtol=0, atol=3e-8)
  np.testing.assert_allclose(desc.solimp[0], [.72, .93, .011, .63, 2.15],
                             rtol=0, atol=1.1e-7)
  np.testing.assert_allclose(desc.friction[0], [.37, .29, .23, .19, .11],
                             rtol=0, atol=3e-8)
  assert float(desc.margin[0]) == pytest.approx(.029)


@pytest.mark.parametrize("profile,integrator", [
    ("integrated_rk4_v1", mujoco.mjtIntegrator.mjINT_RK4),
    ("integrated_implicit_v1", mujoco.mjtIntegrator.mjINT_IMPLICIT),
])
def test_override_admission_for_rk4_and_implicit_profiles(profile, integrator):
  model = _contact_model("dynamic", 3, "elliptic")
  _set_override(model)
  model.opt.integrator = integrator
  admitted = validate_stepping_profile(model, profile=profile)
  assert admitted.name == profile
  assert admitted.model_fingerprint


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("profile,integrator", [
    ("integrated_rk4_v1", mujoco.mjtIntegrator.mjINT_RK4),
    ("integrated_implicit_v1", mujoco.mjtIntegrator.mjINT_IMPLICIT),
])
def test_native_override_executes_with_rk4_and_implicit_profiles(profile, integrator):
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _contact_model("dynamic", 3, "elliptic")
  _set_override(model)
  model.opt.integrator = integrator
  sim = MetalSimulation(model, batch_size=1, profile=profile)
  qpos = np.asarray(model.qpos0, np.float32)[None].copy()
  qvel = np.zeros((1, model.nv), np.float32)
  qvel[0, :3] = [.3, -.2, -1.1]
  qvel[0, 3:6] = [1.3, -2.1, -.5]
  sim.reset(qpos=qpos, qvel=qvel)
  system = sim.assembled_system(recompute=True)
  desc = sim._coupled_constraints.descriptor
  np.testing.assert_allclose(desc.solref[0], [.013, .79], rtol=0, atol=3e-8)
  np.testing.assert_allclose(desc.solreffriction[0], [.013, .79],
                             rtol=0, atol=3e-8)
  assert float(torch.max(torch.abs(system["qfrc_constraint"])).item()) > 1e-5
  sim.step(3)
  snapshot = sim.snapshot()
  assert np.all(np.isfinite(snapshot["device"].qacc))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_override_mixed_rigid_flex_rows_and_qacc():
  """One scene exercises overridden rigid and flex contact row families."""
  import torch
  from mujoco_metal.flex_contact import lower_flex_contacts
  from mujoco_metal.simulation import MetalSimulation

  model = _mixed_rigid_flex_model()
  _set_override(model, margin=.021)
  rigid_desc = lower_coupled_constraints(model)
  flex_desc = lower_flex_contacts(model)
  assert rigid_desc.npairs == 1
  assert flex_desc.slot_count > 0
  np.testing.assert_allclose(rigid_desc.contact_solreffriction[0],
                             [.013, .79], rtol=0, atol=3e-8)
  np.testing.assert_allclose(flex_desc.solref[0], [.013, .79],
                             rtol=0, atol=3e-8)
  np.testing.assert_allclose(flex_desc.friction[0], [.37, .29, .23, .19, .11],
                             rtol=0, atol=3e-8)
  assert float(flex_desc.margin[0]) == pytest.approx(.021)

  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(model.qpos0, np.float32)[None].copy()
  qvel = np.zeros((1, model.nv), np.float32)
  qvel[0, :6] = [.2, .1, -.4, .3, -.2, .1]
  qvel[0, 6:] = np.linspace(-.03, .04, model.nv - 6, dtype=np.float32)
  sim.reset(qpos=qpos, qvel=qvel)
  native = sim.assembled_system(recompute=True)
  flex_bundle = sim._coupled_constraints._flex_contact_current
  assert flex_bundle is not None
  active_flex = flex_bundle["contact_result"]["active"]
  assert bool(torch.any(active_flex).item())
  assert float(native["contact_mask"][0, 0].item()) > .5
  assert float(torch.max(torch.abs(native["qfrc_constraint"])).item()) > 1e-5
  assert float(torch.max(torch.abs(native["qacc"])).item()) > 1e-5

  # CPU supports the rigid contact half of this mixed scene. Flex contact
  # rows are native-only, so compare the independent pinned rigid-force oracle
  # at the same public float32 initial state and check its nonzero response.
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos[0].astype(np.float64)
  cpu.qvel[:] = qvel[0].astype(np.float64)
  mujoco.mj_forward(model, cpu)
  assert cpu.ncon > 0
  contact_force = np.zeros(6)
  mujoco.mj_contactForce(model, cpu, 0, contact_force)
  assert np.linalg.norm(contact_force[1:3]) > 1e-5
  np.testing.assert_allclose(
      native["contact_wrench"][0, 0].detach().cpu().numpy(),
      contact_force, rtol=2e-3, atol=.2)
  np.testing.assert_allclose(native["qacc"][0, :6].detach().cpu().numpy(),
                             cpu.qacc[:6], rtol=2e-2, atol=.5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("pair_mode", ["explicit", "dynamic"])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_native_global_override_matches_contact_rows_force_and_replay(
    pair_mode, cone, condim):
  """Exercise changed override parameters in real coupled rows and trajectory."""
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _contact_model(pair_mode, condim, cone)
  _set_override(model)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(model.qpos0, np.float32)[None].copy()
  qvel = np.zeros((1, model.nv), np.float32)
  qvel[0, :3] = [.3, -.2, -1.1]
  qvel[0, 3:6] = [1.3, -2.1, -.5]
  sim.reset(qpos=qpos, qvel=qvel)
  assembly = sim.assembled_system(recompute=True)

  reference = mujoco.MjData(model)
  reference.qpos[:] = qpos[0]
  reference.qvel[:] = qvel[0]
  mujoco.mj_forward(model, reference)
  assert reference.ncon == 1
  assert np.any(np.abs(reference.efc_J) > 0)
  assert np.any(np.abs(reference.efc_R) > 0)
  assert np.any(np.abs(reference.efc_aref) > 0)

  desc = sim._coupled_constraints.descriptor
  packed = np.asarray(desc.contact_condim_packed)
  gpu_row = int(desc.nr_joint) + int(packed[1])
  cpu_row = next(r for r in range(reference.nefc)
                 if int(reference.efc_type[r]) in (
                     int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                     int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                     int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)))
  rows = (1 if condim == 1 else
          2 * (condim - 1) if cone == "pyramidal" else condim)
  gpu_rows = np.arange(gpu_row, gpu_row + rows)
  cpu_rows = np.arange(cpu_row, cpu_row + rows)
  native_j = assembly["J"][0, gpu_rows].detach().cpu().numpy()
  np.testing.assert_allclose(native_j,
      np.asarray(reference.efc_J).reshape(reference.nefc, model.nv)[cpu_rows],
      atol=4e-6, rtol=2e-5)
  np.testing.assert_allclose(assembly["R"][0, gpu_rows].detach().cpu().numpy(),
                             reference.efc_R[cpu_rows], atol=2e-5, rtol=3e-5)
  np.testing.assert_allclose(assembly["ar"][0, gpu_rows].detach().cpu().numpy(),
                             reference.efc_aref[cpu_rows], atol=2e-3, rtol=3e-4)
  assert float(torch.max(torch.abs(assembly["J"][0, gpu_rows])).item()) > 0
  np.testing.assert_allclose(assembly["qfrc_constraint"][0].detach().cpu().numpy(),
                             reference.qfrc_constraint, rtol=7e-4, atol=1e-1)
  np.testing.assert_allclose(assembly["qacc"][0].detach().cpu().numpy(),
                             reference.qacc, rtol=7e-4, atol=3e-2)
  assert float(torch.max(torch.abs(assembly["qfrc_constraint"][0])).item()) > 0
  native_wrench = assembly["contact_wrench"][0, 0].detach().cpu().numpy()
  cpu_wrench = np.zeros(6)
  mujoco.mj_contactForce(model, reference, 0, cpu_wrench)
  np.testing.assert_allclose(native_wrench, cpu_wrench, rtol=1e-3, atol=.1)
  if condim > 1:
    assert np.linalg.norm(native_wrench[1:3]) > 1e-4

  baseline_model = _contact_model(pair_mode, condim, cone)
  baseline = MetalSimulation(
      baseline_model, batch_size=1, profile="integrated_euler_v1")
  baseline.reset(qpos=qpos, qvel=qvel)
  baseline_assembly = baseline.assembled_system(recompute=True)
  assert float(torch.max(torch.abs(
      assembly["R"][0, gpu_rows] - baseline_assembly["R"][0, gpu_rows])).item()) > 1e-5
  assert float(torch.max(torch.abs(
      assembly["ar"][0, gpu_rows] - baseline_assembly["ar"][0, gpu_rows])).item()) > 1e-4
  assert float(torch.max(torch.abs(
      assembly["qfrc_constraint"][0]
      - baseline_assembly["qfrc_constraint"][0])).item()) > 1e-4

  checkpoint = sim.snapshot()
  sim.step(8)
  first = sim.snapshot()
  sim.restore(checkpoint)
  sim.step(8)
  replay = sim.snapshot()
  for field in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(
        getattr(replay["device"], field), getattr(first["device"], field))
  np.testing.assert_array_equal(
      replay["native_state"]["qacc_warmstart"],
      first["native_state"]["qacc_warmstart"])
  np.testing.assert_array_equal(replay["warmstart"], first["warmstart"])
  assert np.all(np.isfinite(replay["device"].qacc))
  baseline.step(8)
  base_trajectory = baseline.state.snapshot()
  assert (np.max(np.abs(first["device"].qpos - base_trajectory.qpos)) > 0
          or np.max(np.abs(first["device"].qvel - base_trajectory.qvel)) > 0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_override_mutation_rejects_atomically_and_lifecycle_rebuilds():
  """Mutable source options cannot silently detach from compiled rows."""
  from mujoco_metal.simulation import MetalSimulation

  model = _contact_model("dynamic", 4, "elliptic")
  _set_override(model)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(model.qpos0, np.float32)[None].copy()
  qvel = np.zeros((1, model.nv), np.float32)
  qvel[0, :3] = [.3, -.2, -1.1]
  qvel[0, 3:6] = [1.3, -2.1, -.5]
  sim.reset(qpos=qpos, qvel=qvel)
  checkpoint = sim.snapshot()
  sim.step(5)
  completed = sim.snapshot()

  source_options = (int(model.opt.enableflags), float(model.opt.o_margin),
                    model.opt.o_solref.copy(), model.opt.o_solimp.copy(),
                    model.opt.o_friction.copy())
  model.opt.o_friction[0] += .1
  state_before = sim._state.snapshot()
  warm_before = sim._coupled_constraints.get_warmstart().copy()
  cache_before = sim._last_coupled
  calls = (lambda: sim.step(1), lambda: sim.assembled_system(),
           lambda: sim.snapshot(), lambda: sim.restore(checkpoint))
  for call in calls:
    with pytest.raises(ValueError, match="changed after compilation"):
      call()
    state_after = sim._state.snapshot()
    for field in ("qpos", "qvel", "qacc", "time", "status"):
      np.testing.assert_array_equal(getattr(state_after, field),
                                    getattr(state_before, field))
    np.testing.assert_array_equal(
        sim._coupled_constraints.get_warmstart(), warm_before)
    assert sim._last_coupled is cache_before

  # Restore the exact compiled options and prove the prior checkpoint remains
  # replayable. Then adopt changed options via the documented atomic rebuild.
  model.opt.enableflags = source_options[0]
  model.opt.o_margin = source_options[1]
  model.opt.o_solref[:] = source_options[2]
  model.opt.o_solimp[:] = source_options[3]
  model.opt.o_friction[:] = source_options[4]
  sim.restore(checkpoint)
  sim.step(5)
  replayed = sim.snapshot()
  for field in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(getattr(replayed["device"], field),
                                  getattr(completed["device"], field))
  np.testing.assert_array_equal(
      replayed["native_state"]["qacc_warmstart"],
      completed["native_state"]["qacc_warmstart"])
  np.testing.assert_array_equal(replayed["warmstart"], completed["warmstart"])

  model.opt.o_friction[0] += .1
  sim.apply_lifecycle(model)
  np.testing.assert_allclose(
      sim._coupled_constraints.descriptor.friction[0],
      np.maximum(model.opt.o_friction, 1e-5), rtol=0, atol=3e-8)
  sim.step(1)
  assert np.all(np.isfinite(sim.snapshot()["device"].qacc))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("pair_mode", ["explicit", "dynamic"])
def test_native_override_margin_gap_activates_positive_distance(pair_mode):
  """Positive-distance rows use overridden margin and retain authored gap."""
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _contact_model(pair_mode, 3, "pyramidal", z=.205)
  _set_override(model, margin=.02)
  desc = lower_coupled_constraints(model)
  assert float(desc.margin[0]) == pytest.approx(.02)
  assert float(desc.gap[0]) == pytest.approx(
      .004 if pair_mode == "explicit" else .008)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, np.float32)[None].copy(),
            qvel=np.zeros((1, model.nv), np.float32))
  assembled = sim.assembled_system(recompute=True)
  assert float(assembled["contact_mask"][0, 0].item()) > .5

  reference = mujoco.MjData(model)
  reference.qpos[:] = model.qpos0
  mujoco.mj_forward(model, reference)
  contact = next(c for c in reference.contact[:reference.ncon]
                 if set(map(int, c.geom)) == {0, 1})
  assert contact.dist > 0
  assert contact.dist < .02
  assert not bool(contact.exclude)
