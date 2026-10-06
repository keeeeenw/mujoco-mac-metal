# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracle tests for pinned inverse-skip/discrete acceleration semantics."""

import numpy as np
import pytest
import os
from types import MethodType, SimpleNamespace


def _model(integrator="Euler"):
  import mujoco
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".01" integrator="{integrator}">
      <flag contact="disable"/>
    </option>
    <worldbody><body pos="0 0 1">
      <joint type="slide" axis="1 0 0" damping=".7"/>
      <geom type="sphere" size=".1" mass="2"/>
    </body></worldbody>
  </mujoco>''')


def test_pinned_euler_inverse_discrete_uses_damper_derivative_and_restores_qacc():
  import mujoco
  model = _model("Euler")
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  data = mujoco.MjData(model)
  data.qpos[:] = .15
  data.qvel[:] = .4
  original_qacc = np.array([1.3], dtype=np.float64)
  data.qacc[:] = original_qacc
  mujoco.mj_forward(model, data)
  # Forward does not own the requested inverse acceleration.
  data.qacc[:] = original_qacc
  mujoco.mj_inverse(model, data)
  assert np.array_equal(data.qacc, original_qacc)

  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  derivative = np.asarray(model.dof_damping, dtype=np.float64)
  qacc_continuous = np.linalg.solve(
      mass, (mass + model.opt.timestep * np.diag(derivative)) @ original_qacc)
  expected = (mass @ qacc_continuous + data.qfrc_bias
              - data.qfrc_passive - data.qfrc_constraint)
  np.testing.assert_allclose(data.qfrc_inverse, expected, rtol=1e-12, atol=1e-12)


def test_inverse_discrete_euler_production_pair_preserves_acceleration_low():
  """Execute the simulation conversion against an independent f64 oracle."""
  import torch
  import mujoco
  from types import SimpleNamespace
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.simulation import MetalSimulation

  class DenseSolve:
    def run_device(self, mass, rhs, *, awake_lists=None, retained=None):
      del awake_lists, retained
      answer = torch.linalg.solve(mass, rhs.unsqueeze(-1)).squeeze(-1)
      return answer, torch.zeros((mass.shape[0],), dtype=torch.int32)

  model = SimpleNamespace(
      nv=1,
      opt=SimpleNamespace(integrator=mujoco.mjtIntegrator.mjINT_EULER,
                          disableflags=0, timestep=0.003),
      dof_damping=np.array([0.73]),
      dof_dampingpoly=np.array([[0.17, 0.035]]),
      jnt_actuatorid=np.array([-1]), dof_jntid=np.array([0]))
  qacc = torch.tensor([[4096.0]], dtype=torch.float32)
  qacc_low = torch.tensor([[0.000244140625]], dtype=torch.float32)
  qvel = torch.tensor([[0.37]], dtype=torch.float32)
  mass = torch.tensor([[[2.75]]], dtype=torch.float32)
  sim = MetalSimulation.__new__(MetalSimulation)
  sim._state = SimpleNamespace(_torch=torch, _device=torch.device("cpu"))
  sim._mjmodel = model
  sim.batch_size = 1
  sim._component_mass_enabled = False
  sim._solver = DenseSolve()
  sim._passive = SimpleNamespace(
      _damp=torch.tensor([0.73], dtype=torch.float32),
      _dpoly=torch.tensor([0.17, 0.035], dtype=torch.float32))
  dynamics = {"mass_matrix": mass}
  velocity = {"dynamics": dynamics, "qvel": qvel}
  record = SimpleNamespace(values={ForwardStage.POS: {"awake_lists": None}})

  actual_hi, actual_low, status = MetalSimulation._inverse_discrete_acceleration(
      sim, record, velocity, qacc, qacc_low=qacc_low)
  speed = abs(float(qvel.item()))
  derivative = 0.73 + 2 * 0.17 * speed + 3 * 0.035 * speed**2
  exact_qacc = float(qacc.item()) + float(qacc_low.item())
  oracle = ((2.75 + 0.003 * derivative) / 2.75) * exact_qacc
  represented = float(actual_hi.item()) + float(actual_low.item())
  assert status.item() == 0
  assert actual_low.item() != 0
  assert abs(represented - oracle) <= 4.0e-7
  assert abs(float(actual_hi.item()) - oracle) > 100 * abs(represented - oracle)


def test_inverse_discrete_mass_pair_product_uses_compiled_component_pair_map():
  """Route the represented pair through the compiled component mass API."""
  import torch
  from types import SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.component_solve import component_mass_matvec_cpu

  sim = MetalSimulation.__new__(MetalSimulation)
  sim._state = SimpleNamespace(_torch=torch)
  sim._component_mass_enabled = True
  sim._component_solver = SimpleNamespace()
  blocks = torch.tensor([[2.0, 0.25, 0.25, 3.0, 4.0]], dtype=torch.float32)
  armature = torch.tensor([[0.125, 0.0, 0.0, 0.0, 0.5]], dtype=torch.float32)
  vector = torch.tensor([[1.25, -2.0, 3.5]], dtype=torch.float32)
  vector_low = torch.tensor([[2.0**-24, -2.0**-22, 2.0**-23]],
                            dtype=torch.float32)
  layout = {
      "component_mass_offsets": np.array([0, 4], dtype=np.int32),
      "component_dof_offsets": np.array([0, 2, 3], dtype=np.int32),
      "component_dofnum": np.array([2, 1], dtype=np.int32),
      "component_dof_ids": np.array([2, 0, 1], dtype=np.int32),
  }
  seen = []
  def run_pair(actual_blocks, vector_hi, actual_low, *, dof_ids, counts,
               tendon_armature_blocks, diagonal_add):
    seen.append((actual_blocks is blocks, vector_hi is vector,
                 actual_low is vector_low, dof_ids, counts,
                 tendon_armature_blocks is armature))
    map_layout = dict(layout, nv=3, ncomponent=2, nnz=5)
    active = np.ones((1, 3), dtype=bool)
    x = vector_hi.numpy().astype(np.float64) + actual_low.numpy().astype(np.float64)
    result = component_mass_matvec_cpu(
        actual_blocks.numpy(), map_layout, x, active_dof=active,
        tendon_armature_blocks=tendon_armature_blocks.numpy())
    hi = torch.tensor(result.astype(np.float32), dtype=torch.float32)
    lo = torch.tensor((result - hi.numpy().astype(np.float64)).astype(np.float32),
                      dtype=torch.float32)
    return hi, lo
  sim._component_solver.run_mass_matvec_pair_device = run_pair
  high, low = MetalSimulation._inverse_discrete_mass_pair_product(
      sim, {"mass_blocks": blocks, "mass_block_layout": layout,
            "tendon_armature_blocks": armature}, vector, vector_low)
  assert len(seen) == 1 and seen[0][:3] == (True, True, True)
  assert seen[0][3:5] == (None, None) and seen[0][5]
  effective = np.array([[3.0, 0.0, 0.25],
                        [0.0, 4.5, 0.0],
                        [0.25, 0.0, 2.125]], dtype=np.float64)
  oracle = effective @ (vector.numpy().astype(np.float64)[0]
                       + vector_low.numpy().astype(np.float64)[0])
  actual = high.numpy().astype(np.float64)[0] + low.numpy().astype(np.float64)[0]
  np.testing.assert_allclose(actual, oracle, rtol=2e-7, atol=2e-7)


def test_component_implicit_discrete_routes_same_pair_through_effective_operator():
  """Do not form ``A*x_hi`` and ``A*x_low`` as separate rounded products."""
  import torch
  import mujoco
  from types import SimpleNamespace
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.simulation import MetalSimulation

  high = torch.tensor([[4096.0, -2048.0]], dtype=torch.float32)
  low = torch.tensor([[2.0**-12, -2.0**-13]], dtype=torch.float32)
  mass_result = torch.tensor([[8.0, -4.0]], dtype=torch.float32)
  mass_low = torch.tensor([[2.0**-14, 2.0**-15]], dtype=torch.float32)
  effective_status = torch.zeros((1,), dtype=torch.int32)
  pair_calls, solve_calls = [], []
  writer = SimpleNamespace(values=torch.zeros((1, 3), dtype=torch.float32))
  writer.clear_device = lambda: None
  effective = SimpleNamespace()
  def apply_pair(mass, x_hi, x_low, edge_values, timestep, **kwargs):
    pair_calls.append((mass, x_hi, x_low, edge_values, timestep))
    return mass_result.clone(), mass_low.clone(), effective_status.clone()
  effective.apply_effective_operator_pair_device = apply_pair
  sim = MetalSimulation.__new__(MetalSimulation)
  sim._state = SimpleNamespace(_torch=torch, _device=torch.device("cpu"))
  sim._mjmodel = SimpleNamespace(opt=SimpleNamespace(
      integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT, timestep=.002))
  sim.batch_size = 1
  sim._component_mass_enabled = True
  sim._effective_implicit = effective
  sim._velocity_derivative_values = writer
  sim._passive = None
  sim._tendons = None
  sim._spatial_tendons = None
  sim._spatial_kin = None
  sim._fluid = None
  sim._actuators = None
  sim._flex = None
  sim._smooth = SimpleNamespace(
      bias_derivative_device=lambda *_a, **_k: None)
  sim._inverse_discrete_mass_pair_solve = (
      lambda dynamics, rhs, rhs_low, awake: (
          solve_calls.append((rhs.clone(), rhs_low.clone(), awake))
          or (rhs.clone(), rhs_low.clone(), torch.zeros((1,), dtype=torch.int32))))
  dynamics = {"mass_blocks": torch.ones((1, 3), dtype=torch.float32)}
  velocity = {"dynamics": dynamics, "qvel": torch.zeros((1, 2), dtype=torch.float32)}
  awake = {"dof_ids": torch.tensor([[0, 1]], dtype=torch.int32),
           "counts": torch.tensor([[0, 0, 2]], dtype=torch.int32)}
  record = SimpleNamespace(
      qpos=torch.zeros((1, 0), dtype=torch.float32),
      values={ForwardStage.POS: {"awake_lists": awake}})
  actual_hi, actual_low, status = MetalSimulation._inverse_discrete_acceleration(
      sim, record, velocity, high, qacc_low=low)
  assert len(pair_calls) == len(solve_calls) == 1
  assert pair_calls[0][1] is high and pair_calls[0][2] is low
  torch.testing.assert_close(solve_calls[0][0], mass_result)
  torch.testing.assert_close(solve_calls[0][1], mass_low)
  torch.testing.assert_close(actual_hi, mass_result)
  torch.testing.assert_close(actual_low, mass_low)
  assert status.item() == 0


def test_inverse_fd_charges_discrete_pair_operator_scratch():
  import mujoco
  from types import SimpleNamespace
  from mujoco_metal.finite_difference import (
      _inverse_discrete_pair_workspace_bytes)
  bit = int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  sim = SimpleNamespace(
      batch_size=2,
      _mjmodel=SimpleNamespace(nv=3, opt=SimpleNamespace(enableflags=bit)))
  assert _inverse_discrete_pair_workspace_bytes(sim) == 2 * 3 * 32 * 4
  sim._mjmodel.opt.enableflags = 0
  assert _inverse_discrete_pair_workspace_bytes(sim) == 0


def test_pinned_euler_inverse_discrete_respects_eulerdamp_disable():
  import mujoco
  model = _model("Euler")
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
  data = mujoco.MjData(model)
  data.qpos[:] = -.2
  data.qvel[:] = -.35
  original_qacc = np.array([-.8], dtype=np.float64)
  data.qacc[:] = original_qacc
  mujoco.mj_forward(model, data)
  data.qacc[:] = original_qacc
  mujoco.mj_inverse(model, data)
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  expected = mass @ original_qacc + data.qfrc_bias - data.qfrc_passive - data.qfrc_constraint
  np.testing.assert_allclose(data.qfrc_inverse, expected, rtol=1e-12, atol=1e-12)


def test_pinned_euler_inverse_discrete_uses_compiled_polynomial_damping():
  import mujoco
  model = _model("Euler")
  model.dof_damping[0] = .25
  model.dof_dampingpoly[0] = [.3, .07]
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  data = mujoco.MjData(model)
  data.qpos[:] = -.12
  data.qvel[:] = -.4
  qacc = np.array([-.8], dtype=np.float64)
  data.qacc[:] = qacc
  mujoco.mj_forward(model, data)
  data.qacc[:] = qacc
  mujoco.mj_inverse(model, data)
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  speed = abs(float(data.qvel[0]))
  derivative = (.25 + 2 * .3 * speed + 3 * .07 * speed * speed)
  continuous = np.linalg.solve(
      mass, (mass + model.opt.timestep * np.diag([derivative])) @ qacc)
  expected = (mass @ continuous + data.qfrc_bias - data.qfrc_passive
              - data.qfrc_constraint)
  np.testing.assert_allclose(data.qfrc_inverse, expected, rtol=1e-12, atol=1e-12)


def test_pinned_euler_inverse_discrete_includes_joint_actuator_damping_inheritance():
  import mujoco
  from mujoco_metal.model import actuator_joint_inheritance
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".02" integrator="Euler"/>
    <worldbody><body><joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1.7"/></body></worldbody>
    <actuator><motor joint="slide" gear="2" damping=".25"/></actuator></mujoco>''')
  model.dof_damping[0] = .15
  model.dof_dampingpoly[0] = [.08, .03]
  model.actuator_dampingpoly[0] = [.04, .02]
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  # Keep the oracle independent of MetalPassive's lowered constants while
  # checking the pinned gear-squared inheritance at the same compiled joint.
  expected_inherit = model.actuator_gear[0, 0] ** 2
  data = mujoco.MjData(model)
  data.qpos[:] = .1
  data.qvel[:] = -.35
  qacc = np.array([.7], dtype=np.float64)
  mujoco.mj_forward(model, data)
  data.qacc[:] = qacc
  mujoco.mj_inverse(model, data)
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  speed = abs(float(data.qvel[0]))
  derivative = (
      model.dof_damping[0] + expected_inherit * model.actuator_damping[0]
      + 2 * (model.dof_dampingpoly[0, 0]
             + expected_inherit * model.actuator_dampingpoly[0, 0]) * speed
      + 3 * (model.dof_dampingpoly[0, 1]
             + expected_inherit * model.actuator_dampingpoly[0, 1]) * speed**2)
  expected_continuous = np.linalg.solve(
      mass, (mass + model.opt.timestep * np.diag([derivative])) @ qacc)
  expected = (mass @ expected_continuous + data.qfrc_bias - data.qfrc_passive
              - data.qfrc_constraint)
  np.testing.assert_allclose(data.qfrc_inverse, expected, rtol=2e-11, atol=2e-11)
  arm, damp, poly = actuator_joint_inheritance(model)
  assert np.isclose(damp[0], expected_inherit * model.actuator_damping[0])
  np.testing.assert_allclose(
      poly[0], expected_inherit * model.actuator_dampingpoly[0], atol=0, rtol=0)


@pytest.mark.parametrize("integrator", ["implicit", "implicitfast"])
def test_pinned_implicit_inverse_discrete_maps_acceleration_through_qderiv(integrator):
  import mujoco
  model = _model(integrator)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  data = mujoco.MjData(model)
  data.qpos[:] = .1
  data.qvel[:] = -.25
  original_qacc = np.array([.9], dtype=np.float64)
  mujoco.mj_forward(model, data)
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  # For this scalar slide, qfrc_bias has no velocity derivative and the only
  # qDeriv entry is the negative linear damper derivative.
  data.qacc[:] = original_qacc
  mujoco.mj_inverse(model, data)
  assert np.array_equal(data.qacc, original_qacc)
  continuous = np.linalg.solve(
      mass, (mass + model.opt.timestep * np.diag(model.dof_damping))
      @ original_qacc)
  expected = mass @ continuous + data.qfrc_bias - data.qfrc_passive - data.qfrc_constraint
  np.testing.assert_allclose(data.qfrc_inverse, expected, rtol=2e-10, atol=2e-10)


def test_integrated_profile_admits_discrete_inverse_but_rk4_reports_pinned_error():
  import mujoco
  from mujoco_metal.stepping import validate_stepping_profile

  euler = _model("Euler")
  euler.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  profile = validate_stepping_profile(euler, profile="integrated_euler_v1")
  assert profile.name == "integrated_euler_v1"

  rk4 = _model("RK4")
  rk4.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  with pytest.raises(ValueError, match="unsupported by the RK4 integrator"):
    validate_stepping_profile(rk4, profile="integrated_rk4_v1")


@pytest.mark.parametrize("profile", ["normal_contact_euler_v1",
                                     "friction_contact_euler_v1",
                                     "joint_constraints_euler_v1"])
@pytest.mark.parametrize("flag", ["mjENBL_INVDISCRETE", "mjENBL_FWDINV"])
def test_legacy_contact_profiles_admit_canonical_inverse_rows(profile, flag):
  import mujoco
  from mujoco_metal.stepping import validate_stepping_profile
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option integrator="Euler"><flag contact="disable"/></option>
    <worldbody><body><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
  </mujoco>''')
  model.opt.enableflags |= int(getattr(mujoco.mjtEnableBit, flag))
  lowered = validate_stepping_profile(model, profile=profile)
  assert lowered.name == profile


def test_inverse_skip_legacy_rows_reach_prepared_state_path():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  sim = MetalSimulation.__new__(MetalSimulation)
  sim._contact = object()
  sim._joint_constraints = None
  # Legacy profiles now have a canonical assembly-only row adapter.  The
  # deliberately incomplete instance must pass that removed admission guard
  # and fail only when it reaches the absent prepared state.
  with pytest.raises(AttributeError, match="_state"):
    sim.inverse_skip(mujoco.mjtStage.mjSTAGE_NONE)


def test_legacy_rows_view_preserves_joint_then_contact_canonical_layout():
  import numpy as np
  from types import SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation

  sim = MetalSimulation.__new__(MetalSimulation)
  # Three scalar joint rows followed by two contact candidates (five slots
  # each), each row is [J[nv], R, ar, lo, hi, active].
  packed = np.arange(13 * 7, dtype=np.float32).reshape(1, 13, 7)
  sim._legacy_canonical_rows = packed
  sim.batch_size = 1
  sim._mjmodel = SimpleNamespace(nv=2)
  sim._joint_constraints = SimpleNamespace(descriptor=SimpleNamespace(
      nrow=3, neq=1))
  rows, descriptor = MetalSimulation._legacy_rows_view(sim)
  assert rows["J"].shape == (1, 13, 2)
  assert rows["R"].shape == rows["active"].shape == (1, 13)
  np.testing.assert_array_equal(rows["J"], packed[..., :2])
  np.testing.assert_array_equal(rows["R"], packed[..., 2])
  np.testing.assert_array_equal(rows["ar"], packed[..., 3])
  np.testing.assert_array_equal(rows["lo"], packed[..., 4])
  np.testing.assert_array_equal(rows["hi"], packed[..., 5])
  np.testing.assert_array_equal(rows["active"], packed[..., 6])
  assert descriptor.nr_joint == 3
  assert descriptor.n_eq_rows == 1
  assert descriptor.ncontacts_max == 0


def test_inverse_skip_rejects_invalid_qacc_before_recomputing_any_stage():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  class Tensor:
    def __init__(self, shape):
      self.shape = shape
      self.dtype = "float32"
      self.device = "mps"
    def is_contiguous(self): return True
  torch = SimpleNamespace(Tensor=Tensor, float32="float32")
  sim = MetalSimulation.__new__(MetalSimulation)
  sim._contact = None
  sim._joint_constraints = None
  sim._state = SimpleNamespace(_torch=torch, _device="mps",
                                _qacc=Tensor((1, 1)))
  sim._mjmodel = SimpleNamespace(nv=1)
  sim.batch_size = 1
  sim.prepare_forward_position = lambda *_a, **_kw: pytest.fail(
      "invalid qacc must fail before POS stage writes")
  with pytest.raises(ValueError, match="qacc must be contiguous"):
    sim.inverse_skip(mujoco.mjtStage.mjSTAGE_NONE, qacc=Tensor((1, 2)))


def test_legacy_inverse_snapshots_qacc_before_row_assembly_reuses_its_output(
    monkeypatch):
  """Legacy assembly may clear its qacc output, which can alias the input."""
  torch = pytest.importorskip("torch")
  import mujoco
  import mujoco_metal.inverse_constraints as inverse_constraints
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.simulation import MetalSimulation

  source_qacc = torch.tensor([[1.0]], dtype=torch.float32)
  source_qacc_low = torch.tensor([[8.0e-8]], dtype=torch.float32)
  source_qacc_low_value = float(source_qacc_low.item())
  qvel = torch.tensor([[0.0]], dtype=torch.float32)
  dynamics = {
      # This value is one float32 ULP above the passive force.  The pinned
      # order computes Ma - passive before adding the large bias, preserving
      # the +8 increment.  Adding bias to Ma first loses it at 1e8 scale.
      "mass_matrix": torch.tensor([[[100000008.0]]], dtype=torch.float32),
      "qfrc_bias_low": torch.zeros((1, 1), dtype=torch.float32),
      "poses": {},
  }
  velocity = {
      "dynamics": dynamics,
      "qvel": qvel,
      "qfrc_bias": torch.tensor([[100000000.0]], dtype=torch.float32),
      "qfrc_passive": torch.tensor([[100000000.0]], dtype=torch.float32),
      "status": torch.zeros((1,), dtype=torch.int32),
  }
  record = SimpleNamespace(
      qpos=torch.zeros((1, 1), dtype=torch.float32),
      values={ForwardStage.POS: {"awake_lists": None}})
  observed = {}

  sim = MetalSimulation.__new__(MetalSimulation)
  sim.batch_size = 1
  sim._state = SimpleNamespace(
      _torch=torch, _device=torch.device("cpu"), _qacc=source_qacc,
      _eq_active=None, generation=1)
  sim._last_sensor_qacc = source_qacc
  sim._last_sensor_qacc_version = int(source_qacc._version)
  sim._last_sensor_qacc_generation = 1
  sim._last_sensor_qacc_low = source_qacc_low
  sim._mjmodel = SimpleNamespace(
      nv=1, opt=SimpleNamespace(enableflags=0))
  sim._component_mass_enabled = False
  sim._coupled_constraints = None
  sim._contact = object()
  sim._joint_constraints = None
  sim._passive = None
  sim._body_wrench = None
  sim._sensors = None
  sim._sensordata = None
  sim._energy = None
  sim._forward_stages = SimpleNamespace(consume=lambda *_args, **_kwargs: velocity)
  sim.validate_forward_stage_record = lambda current, *_args, **_kwargs: current
  def assemble_legacy_rows(*_args):
    source_qacc.zero_()
    source_qacc_low.zero_()
    return source_qacc, {"nr_joint": 1}
  sim._assemble_legacy_inverse_rows = assemble_legacy_rows

  def inverse_force(_rows, qacc, _descriptor, *, qacc_low=None,
                    return_low=False):
    observed["qacc"] = qacc.clone()
    observed["qacc_low"] = (None if qacc_low is None else qacc_low.clone())
    high = torch.zeros_like(qacc)
    return (high, torch.zeros_like(high)) if return_low else high

  monkeypatch.setattr(inverse_constraints, "inverse_constraint_force",
                      inverse_force)
  result = sim.inverse_skip(mujoco.mjtStage.mjSTAGE_ACC, record=record)
  torch.testing.assert_close(observed["qacc"], torch.tensor([[1.0]]))
  torch.testing.assert_close(observed["qacc_low"],
                             torch.tensor([[source_qacc_low_value]]))
  represented_force = (result["qfrc_inverse"].double()
                      + result["qfrc_inverse_low"].double())
  torch.testing.assert_close(
      represented_force,
      torch.tensor([[100000008.0 + 100000008.0 * source_qacc_low_value]],
                   dtype=torch.float64), rtol=0, atol=0)

  # A separately supplied high-only acceleration has no relation to the
  # simulation-owned low plane, even if the state still advertises one.
  explicit_qacc = torch.tensor([[2.0]], dtype=torch.float32)
  sim.inverse_skip(mujoco.mjtStage.mjSTAGE_ACC, record=record,
                   qacc=explicit_qacc)
  torch.testing.assert_close(observed["qacc"], explicit_qacc)
  assert observed["qacc_low"] is None


def test_implicit_inverse_failure_restores_actuator_and_passive_scratch(monkeypatch):
  import mujoco
  import mujoco_metal.simulation as simulation_module
  from mujoco_metal.simulation import MetalSimulation

  class Buffer:
    def __init__(self, *values):
      self.values = list(values)
      self.shape = (len(values),)

    def clone(self):
      return Buffer(*self.values)

    def fill_(self, value):
      self.values[:] = [value] * len(self.values)

    def copy_(self, other):
      self.values[:] = other.values
      return self

  def clone_tree(value):
    if isinstance(value, Buffer):
      return value.clone()
    if isinstance(value, dict):
      return {key: clone_tree(item) for key, item in value.items()}
    return value

  monkeypatch.setattr(simulation_module, "_clone_system_dict", clone_tree)

  force = Buffer(1.0, 2.0)
  passive_force = Buffer(3.0)
  act_dot = Buffer(4.0)
  act_vel = Buffer(5.0)
  derivative = Buffer(6.0)
  kin = {"length": Buffer(7.0)}
  fake = SimpleNamespace(
      _mjmodel=SimpleNamespace(opt=SimpleNamespace(
          integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT)),
      _actuators=SimpleNamespace(_ws={"force": force, "act_dot": act_dot,
                                     "velocity_derivative": derivative}),
      _passive=SimpleNamespace(_force=passive_force),
      _last_actuation_kin=kin,
      _actuator_velocity_derivative=derivative,
      _act_dot=act_dot, _act_vel=act_vel,
      _sen_act_force=Buffer(8.0),
      _sen_qfrc_act=Buffer(9.0))

  def fail_after_mutating(self, _record, _velocity, _qacc):
    self._actuators._ws["force"].fill_(20.0)
    self._passive._force.fill_(21.0)
    self._act_dot.fill_(22.0)
    self._act_vel.fill_(23.0)
    self._actuator_velocity_derivative.fill_(24.0)
    self._last_actuation_kin = {"length": Buffer(-1.0)}
    self._sen_act_force.fill_(25.0)
    self._sen_qfrc_act.fill_(26.0)
    raise RuntimeError("synthetic inverse derivative failure")

  fake._inverse_discrete_acceleration = MethodType(fail_after_mutating, fake)
  fake._snapshot_inverse_actuator_scratch = MethodType(
      MetalSimulation._snapshot_inverse_actuator_scratch, fake)
  fake._restore_inverse_actuator_scratch = MethodType(
      MetalSimulation._restore_inverse_actuator_scratch, fake)
  with pytest.raises(RuntimeError, match="synthetic inverse derivative failure"):
    MetalSimulation._inverse_discrete_with_preserved_actuator_scratch(
        fake, object(), object(), object())
  assert force.values == [1.0, 2.0]
  assert passive_force.values == [3.0]
  assert act_dot.values == [4.0]
  assert act_vel.values == [5.0]
  assert derivative.values == [6.0]
  assert fake._sen_act_force.values == [8.0]
  assert fake._sen_qfrc_act.values == [9.0]
  assert fake._last_actuation_kin is kin


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires explicit native inverse-skip qualification")
def test_native_inverse_skip_discrete_matches_pinned_euler():
  import mujoco
  from mujoco_metal import MetalSimulation

  model = _model("Euler")
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  data = mujoco.MjData(model)
  data.qpos[:] = .15
  data.qvel[:] = .4
  qacc = np.array([1.3], dtype=np.float32)
  mujoco.mj_forward(model, data)
  data.qacc[:] = qacc
  mujoco.mj_inverse(model, data)
  expected = np.array(data.qfrc_inverse, dtype=np.float32)

  sim = MetalSimulation(
      model, batch_size=1, qpos=np.asarray(data.qpos, dtype=np.float32)[None],
      qvel=np.asarray(data.qvel, dtype=np.float32)[None],
      profile="integrated_euler_v1")
  result = sim.inverse_skip(qacc=sim.state._torch.as_tensor(
      qacc[None], dtype=sim.state._torch.float32, device=sim.state._device))
  np.testing.assert_allclose(result["qfrc_inverse"].cpu().numpy()[0],
                             expected, rtol=3e-4, atol=3e-5)
