# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Observational coupled assembly preserves state, borrowed owners and replay."""
from dataclasses import fields, is_dataclass
import os

import mujoco
import numpy as np
import pytest


def _assert_snapshot_bytes(before, after, *, replay=False):
  if isinstance(before, np.ndarray):
    assert before.dtype == after.dtype and before.shape == after.shape
    assert before.tobytes() == after.tobytes()
  elif isinstance(before, dict):
    assert before.keys() == after.keys()
    for name in before:
      _assert_snapshot_bytes(before[name], after[name], replay=replay)
  elif is_dataclass(before):
    for field in fields(before):
      if replay and field.name == "row_reset_epoch":
        continue  # restore intentionally invalidates borrowed row records
      _assert_snapshot_bytes(getattr(before, field.name), getattr(after, field.name), replay=replay)
  else:
    assert before == after


def _model(sparse):
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001" integrator="Euler" solver="PGS" iterations="80">
      <flag energy="enable"/>
    </option>
    <worldbody><geom type="plane" size="2 2 .1" condim="1"/>
      <body pos="0 0 .099"><joint name="vertical" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1" condim="1"/></body>
    </worldbody>
    <actuator><general joint="vertical" dyntype="filterexact" dynprm=".05"/></actuator>
    <sensor><jointpos joint="vertical"/><jointvel joint="vertical"/></sensor>
    </mujoco>''')
  model.opt.jacobian = (mujoco.mjtJacobian.mjJAC_SPARSE if sparse
                        else mujoco.mjtJacobian.mjJAC_DENSE)
  if sparse:
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  return model


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native GPU opt-in")
@pytest.mark.parametrize("sparse", (False, True))
@pytest.mark.parametrize("failure", (False, True))
def test_observational_assembly_preserves_checkpoint_and_borrowed_results(
    sparse, failure, monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  model = _model(sparse)
  sim = MetalSimulation(model, 2, profile=("integrated_implicitfast_v1" if sparse
                                           else "integrated_euler_v1"))
  assert sim._component_mass_enabled is sparse
  qpos = np.array([[0.], [-.0005]], dtype=np.float32)
  qvel = np.array([[-.02], [.01]], dtype=np.float32)
  act = np.array([[.2], [.3]], dtype=np.float32)
  ctrl = np.array([[.4], [.5]], dtype=np.float32)
  force = np.array([[.125], [-.125]], dtype=np.float32)
  sim.reset(qpos=qpos, qvel=qvel, act=act)
  assert sim.step(ctrl=ctrl, qfrc_applied=force).cpu().tolist() == [0, 0]
  # Capture real producer workspaces, not the separately owned accepted-step
  # history returned by a cached assembled_system call.
  borrowed = sim.assembled_system(recompute=True)
  sim.step1()
  record = sim._step1_record
  before = sim.snapshot()
  # The control is valid and would replace the held value before malformed
  # force validation. Even that partial input preparation must roll back.
  with pytest.raises(ValueError):
    sim.query_assembled_system(ctrl=ctrl * .25,
                               qfrc_applied=np.zeros((2, 2), dtype=np.float32))
  _assert_snapshot_bytes(before, sim.snapshot())
  owners = {key: (value, value.clone()) for key, value in borrowed.items()
            if isinstance(value, torch.Tensor)}
  # Independent pinned forward results at the actual represented current
  # state, with the query's inputs (not the restored held inputs).
  state = before["device"]
  words = state.precision_words.astype(np.float64)
  positions = state.qpos.astype(np.float64) + words[:, :1] + words[:, 1:2]
  velocities = state.qvel.astype(np.float64) + words[:, 2:3] + words[:, 3:4]
  references = []
  for row in range(2):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:] = positions[row], velocities[row], state.act[row]
    data.ctrl[:] = ctrl[row] * .5
    data.qfrc_applied[:] = force[row] * 2
    data.qacc_warmstart[:] = before["native_state"]["qacc_warmstart"][row]
    mujoco.mj_forward(model, data)
    assert data.ncon == 1 and np.linalg.norm(data.qfrc_constraint) > .01
    references.append(data)
  cc = sim._coupled_constraints
  original = cc.run_device
  calls = []
  def guarded(*args, **kwargs):
    calls.append(True)
    result = original(*args, **kwargs)
    if failure:
      raise RuntimeError("injected post-assembly failure")
    return result
  with monkeypatch.context() as context:
    context.setattr(cc, "run_device", guarded)
    def forbidden(*args, **kwargs):
      raise AssertionError("observational query fell back to CPU physics")
    for name in ("mj_forward", "mj_step", "mj_step1", "mj_step2", "mj_fwdPosition",
                 "mj_fwdVelocity", "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint"):
      context.setattr(mujoco, name, forbidden)
    if failure:
      with pytest.raises(RuntimeError, match="post-assembly"):
        sim.query_assembled_system(ctrl=ctrl * .5, qfrc_applied=force * 2)
    else:
      output = sim.query_assembled_system(ctrl=ctrl * .5, qfrc_applied=force * 2)
      assert output["status"].cpu().tolist() == [0, 0]
      assert output["contact_mask"].sum().item() > 0
      assert torch.linalg.vector_norm(output["qfrc_constraint"]).item() > .01
  assert calls, "query did not execute native assembly"
  if not failure:
    for field in ("qacc", "qfrc_constraint"):
      np.testing.assert_allclose(output[field].cpu().numpy(),
                                 np.stack([getattr(data, field) for data in references]),
                                 rtol=3e-5, atol=3e-5)
  _assert_snapshot_bytes(before, sim.snapshot())
  assert sim._step1_record is record
  sim.validate_forward_stage_record(record, "VEL")
  for name, (owner, saved) in owners.items():
    assert borrowed[name] is owner
    assert torch.equal(owner, saved), name
    if not failure:
      assert output[name] is not owner
  if not failure:
    owned = {key: value.clone() for key, value in output.items()
             if isinstance(value, torch.Tensor)}
  assert sim.step2(ctrl=ctrl, qfrc_applied=force).cpu().tolist() == [0, 0]
  assert sim.step(2, ctrl=ctrl, qfrc_applied=force).cpu().tolist() == [0, 0]
  expected = sim.snapshot()
  sim.restore(before)
  sim.step1()
  assert sim.step2(ctrl=ctrl, qfrc_applied=force).cpu().tolist() == [0, 0]
  assert sim.step(2, ctrl=ctrl, qfrc_applied=force).cpu().tolist() == [0, 0]
  _assert_snapshot_bytes(expected, sim.snapshot(), replay=True)
  if not failure:
    for key, value in owned.items():
      assert torch.equal(output[key], value), key
