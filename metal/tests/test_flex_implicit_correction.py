# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source-derived flex stiffness solve; integration gates remain separate."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_implicit import FlexImplicitCorrection, OPERATOR_NAMES


def _model(integrator):
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".01" integrator="{integrator}"/>
    <worldbody><body><joint type="slide" axis="1 0 0"/>
    <joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
    <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')


def _problem():
  # Two material modes with distinct damping cannot share a global coefficient.
  vi = np.array([1.,-2.,.4]); vb = np.array([.2,.3,-.6])
  ki = -7*np.outer(vi,vi); kb = 5*np.outer(vb,vb)
  ops = dict(zip(OPERATOR_NAMES,(ki,.03*ki,kb,.08*kb)))
  h = .01
  base = np.array([[2.,.1,0],[.1,1.,.02],[0,.02,.7]])
  velocity = np.array([.2,-.4,.1]); force = np.array([.3,.4,-.2])
  effective = base-h*h*ki-h*.03*ki+h*h*kb+h*.08*kb
  rhs = force+h*(ki-kb)@velocity
  return base,force,velocity,ops,effective,rhs


def test_stiffness_equation_requires_both_velocity_and_damping_corrections():
  base,force,velocity,ops,effective,rhs = _problem()
  corrected = np.linalg.solve(effective,rhs)
  uncorrected = np.linalg.solve(base,force)
  assert np.max(np.abs(corrected-uncorrected)) > .05
  np.testing.assert_allclose(effective@corrected,rhs,atol=1e-14)
  assert np.linalg.eigvalsh(effective).min() > 0


def test_flex_correction_validation_precedes_device_setup():
  with pytest.raises(ValueError,match="implicit or implicitfast"):
    FlexImplicitCorrection(_model("Euler"))
  with pytest.raises(ValueError,match="indexing capacity"):
    FlexImplicitCorrection(_model("implicit"),1 << 30)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",reason="native opt-in")
@pytest.mark.parametrize("integrator",["implicit","implicitfast"])
def test_native_flex_stiffness_correction_and_disabled_world(integrator):
  import torch
  model = _model(integrator)
  program = FlexImplicitCorrection(model,3)
  base,force,velocity,ops,effective,rhs = _problem()
  def tensor(value):
    return torch.tensor(np.broadcast_to(value,(3,*value.shape)).copy(),
                        dtype=torch.float32,device="mps")
  initial = np.linalg.solve(base,force)
  kwargs = dict(preconditioner=tensor(base),base_matrix=tensor(base),
      qfrc_total=tensor(force),qvel=tensor(velocity),initial_qacc=tensor(initial),
      operators={name:tensor(value) for name,value in ops.items()},
      enabled=torch.tensor([True,False,True],device="mps"))
  expected = np.stack([np.linalg.solve(effective,rhs),initial,np.linalg.solve(effective,rhs)])
  pointer = None
  for _ in range(2):
    result = program.run_device(**kwargs)
    np.testing.assert_array_equal(result["status"].cpu().numpy(),0)
    np.testing.assert_allclose(result["qacc"].cpu().numpy(),expected,atol=3e-6,rtol=2e-5)
    counts = result["iterations"].cpu().numpy()
    assert 0 < counts[0] <= 50 and counts[1] == 0 and counts[2] == counts[0]
    if pointer is not None:
      assert pointer == result["qacc"].data_ptr()
    pointer = result["qacc"].data_ptr()
  # Bad preconditioner in one enabled world cannot corrupt its neighbor.
  bad_preconditioner = kwargs["preconditioner"].clone()
  bad_preconditioner[2].fill_(float("nan"))
  kwargs["preconditioner"] = bad_preconditioner
  failed = program.run_device(**kwargs)
  np.testing.assert_allclose(failed["qacc"].cpu().numpy()[0],expected[0],atol=3e-6,rtol=2e-5)
  np.testing.assert_array_equal(failed["qacc"].cpu().numpy()[2],kwargs["initial_qacc"].cpu().numpy()[2])
  np.testing.assert_array_equal(failed["status"].cpu().numpy(),[0,0,1])
  # Replacing the failed input and using the reusable default mask restores
  # all worlds, without retaining a failed status or an old enabled mask.
  kwargs["preconditioner"] = tensor(base)
  kwargs["enabled"] = None
  restored = program.run_device(**kwargs)
  np.testing.assert_array_equal(restored["status"].cpu().numpy(),0)
  np.testing.assert_allclose(restored["qacc"].cpu().numpy(),
      np.broadcast_to(np.linalg.solve(effective,rhs),(3,3)),atol=3e-6,rtol=2e-5)
