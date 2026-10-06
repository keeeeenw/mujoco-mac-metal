"""Executed pinned CPU warning/reset contract, independent of GPU opt-in."""
import mujoco
import numpy as np
import pytest

@pytest.mark.parametrize("autoreset",[True,False])
def test_pinned_cpu_checkacc_qpos_warning_and_repeated_failure(autoreset):
  model=mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><body><joint type="slide" axis="1 0 0"/>'
      '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
  if not autoreset:
    model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
  data=mujoco.MjData(model);data.qpos[0]=.3
  mujoco.mj_forward(model,data)
  warning=int(mujoco.mjtWarning.mjWARN_BADQACC)
  for call in (1,2):
    data.qacc[0]=np.nan
    mujoco.mj_checkAcc(model,data)
    np.testing.assert_array_equal(data.qpos,[0.] if autoreset else [.3])
    assert int(data.warning[warning].number)==(1 if autoreset else 2*call)
    assert int(data.warning[warning].lastinfo)==0
