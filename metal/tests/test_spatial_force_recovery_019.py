"""Spatial spring/damper/armature through native post-forward recovery."""
import os
import mujoco
import numpy as np
import pytest


def _model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep=".001"/>
    <worldbody><site name="anchor" pos=".4 .2 .3"/>
      <body pos=".1 0 .05"><joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
        <site name="tip" pos=".01 .02 .03"/>
      </body></worldbody>
    <tendon><spatial stiffness="12" damping=".4" armature=".03" springlength=".25">
      <site site="anchor"/><site site="tip"/>
    </spatial></tendon>
  </mujoco>""")


def test_cpu_spatial_recovery_fixture_has_nonzero_force_and_armature():
  model=_model(); data=mujoco.MjData(model)
  data.qpos[0]=.02; data.qvel[0]=.1
  mujoco.mj_forward(model,data)
  assert abs(float(data.qfrc_passive[0]))>.1
  assert float(model.tendon_armature[0])>.01
  assert np.isfinite(data.qacc).all()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU")!="1",reason="native recovery gate")
@pytest.mark.parametrize("batch",[1,2])
def test_native_spatial_force_flags_and_masked_recovery_match_cpu(batch):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.state_checks import mj_checkAcc
  model=_model()
  qpos=np.array([[.02],[.08]],np.float32)[:batch].copy()
  qvel=np.array([[.1],[-.12]],np.float32)[:batch].copy()
  sim=MetalSimulation(model,batch_size=batch,qpos=qpos,qvel=qvel,
                      profile="integrated_euler_v1")
  assert sim._spatial_tendons._dims.numel()==11+batch
  refs=[]
  for world in range(batch):
    data=mujoco.MjData(model); data.qpos[:]=qpos[world]; data.qvel[:]=qvel[world]
    mujoco.mj_step(model,data); refs.append(data)
  # A healthy public step still takes the all-zero recovery-mask route.
  status=sim.step()
  np.testing.assert_array_equal(status.detach().cpu().numpy(),np.zeros(batch,np.int32))
  before=sim.state.snapshot()
  for world,data in enumerate(refs):
    np.testing.assert_allclose(before.qpos[world],data.qpos,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(before.qvel[world],data.qvel,rtol=2e-5,atol=2e-6)
  assert np.isfinite(before.qacc).all()
  # Then genuinely reset one world; the other world must remain bit-identical.
  bad=torch.zeros((batch,model.nv),dtype=torch.float32,device="mps")
  bad[0,0]=float("nan")
  result=mj_checkAcc(sim,qacc=bad)
  np.testing.assert_array_equal(result["bad_world"].detach().cpu().numpy(),
                                np.arange(batch)==0)
  after=sim.state.snapshot()
  np.testing.assert_array_equal(after.qpos[0],model.qpos0.astype(np.float32))
  np.testing.assert_array_equal(after.qvel[0],np.zeros(model.nv,np.float32))
  assert np.isfinite(after.qacc).all()
  if batch==2:
    for field in ("qpos","qvel","qacc"):
      np.testing.assert_array_equal(getattr(after,field)[1],getattr(before,field)[1])
  # Continued native stepping uses restored tendon kinematics and controls.
  cpu_reset=mujoco.MjData(model)
  for _ in range(3):
    status=sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(),np.zeros(batch,np.int32))
    mujoco.mj_step(model,cpu_reset)
  final=sim.state.snapshot()
  np.testing.assert_allclose(final.qpos[0],cpu_reset.qpos,rtol=2e-5,atol=2e-6)
  np.testing.assert_allclose(final.qvel[0],cpu_reset.qvel,rtol=2e-5,atol=2e-6)
