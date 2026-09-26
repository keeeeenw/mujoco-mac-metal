# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.fluid import InertiaBoxFluidModel
from mujoco_metal.fluid import MetalInertiaBoxFluid


_XML = '''<mujoco model="inertia_box_fluid">
  <option gravity="0 0 0" density="1.7" viscosity=".09" wind=".6 -.2 .35"/>
  <worldbody>
    <body name="free" pos=".2 -.3 1">
      <freejoint/><inertial pos=".08 -.04 .03" mass="1.3" diaginertia=".08 .11 .14"/>
      <geom type="box" size=".12 .08 .1" mass="1.3"/>
      <body name="hinged" pos=".15 .1 .05">
        <joint name="h" type="hinge" axis=".3 .8 -.2"/>
        <geom type="ellipsoid" size=".1 .16 .07" mass=".6"/>
      </body>
    </body>
  </worldbody>
</mujoco>'''


def _states(model):
  qpos = np.array(model.qpos0, dtype=np.float64)
  qpos[3:7] = [.91, .2, -.31, .1]
  qpos[-1] = .48
  qvel = np.linspace(-.7, .9, model.nv)
  return np.stack([qpos, qpos + np.r_[np.zeros(3), -.02, .01, .03, -.01, .16]]), np.stack([qvel, -qvel])


def _oracle(model, qpos, qvel):
  result=[]
  for q, v in zip(qpos, qvel):
    data=mujoco.MjData(model)
    data.qpos[:]=q
    data.qvel[:]=v
    mujoco.mj_forward(model,data)
    result.append(data.qfrc_fluid.copy())
  return np.asarray(result)


def test_inertia_box_viscous_lift_drag_and_wind_match_mujoco():
  model=mujoco.MjModel.from_xml_string(_XML)
  qpos,qvel=_states(model)
  actual=InertiaBoxFluidModel(model).run(qpos,qvel)
  np.testing.assert_allclose(actual,_oracle(model,qpos,qvel),rtol=3e-7,atol=3e-7)
  assert np.linalg.norm(actual)>0


def test_fluid_disable_early_return_and_zero_coefficients():
  spring=int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  damper=int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  model=mujoco.MjModel.from_xml_string(_XML)
  qpos,qvel=_states(model)
  for flags in (spring,damper,spring|damper):
    model.opt.disableflags=flags
    actual=InertiaBoxFluidModel(model).run(qpos,qvel)
    np.testing.assert_allclose(actual,_oracle(model,qpos,qvel),rtol=3e-7,atol=3e-7)
  model.opt.disableflags=0
  model.opt.density=0
  model.opt.viscosity=0
  np.testing.assert_array_equal(InertiaBoxFluidModel(model).run(qpos,qvel),np.zeros((2,model.nv)))


def test_geom_fluid_models_rejected():
  model=mujoco.MjModel.from_xml_string(_XML)
  model.geom_fluid[0,0]=1
  with pytest.raises(ValueError,match='geom-fluid'):
    InertiaBoxFluidModel(model)


def test_state_shape_and_nonfinite_values_rejected():
  model=mujoco.MjModel.from_xml_string(_XML)
  stage=InertiaBoxFluidModel(model)
  with pytest.raises(ValueError,match='shapes'):
    stage.run(np.zeros(model.nq),np.zeros((1,model.nv)))
  qpos,qvel=_states(model)
  qvel[0,0]=np.inf
  with pytest.raises(ValueError,match='finite'):
    stage.run(qpos,qvel)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_inertia_box_fluid_matches_mujoco():
  import torch
  model=mujoco.MjModel.from_xml_string(_XML)
  qpos,qvel=_states(model)
  stage=MetalInertiaBoxFluid(model,batch_size=len(qpos))
  qpos_device=torch.tensor(qpos,dtype=torch.float32,device="mps")
  qvel_device=torch.tensor(qvel,dtype=torch.float32,device="mps")
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  dynamics=MetalSmoothDynamics(load_model(model),batch_size=len(qpos)).run_device(qpos_device,qvel_device)
  actual=stage.run_device(qpos_device,qvel_device,dynamics)
  np.testing.assert_allclose(actual.cpu().numpy(),_oracle(model,qpos,qvel),rtol=2e-5,atol=2e-5)
