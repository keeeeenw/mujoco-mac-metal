"""Native regression for consecutive row-mask helpers without host barriers."""
import os
import numpy as np
import pytest
import mujoco

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_consecutive_row_helpers_bind_each_selector_without_host_barriers():
    import torch
    from mujoco_metal.device_state import DeviceState
    from mujoco_metal.stepping import validate_stepping_profile
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="0 0 0"><flag contact="disable"/></option>
      <worldbody><body><joint type="hinge"/><geom type="sphere" size=".1"/>
      </body></worldbody></mujoco>''')
    state = DeviceState(model, validate_stepping_profile(model), 2, device="mps")
    def tensor(value, dtype=torch.float32):
        return torch.tensor(value, dtype=dtype, device="mps")
    first = tensor([True, False], torch.bool)
    second = tensor([False, True], torch.bool)
    none = tensor([False, False], torch.bool)
    # Physics producers retain this selector across row-local cache repairs.
    state._reset_mask_i32.copy_(tensor([1, 0], torch.int32))
    scratch = tensor([[100., 200.], [300., 400.]])
    target = tensor([[1., 2.], [3., 4.]])
    integers = tensor([[7], [8]], torch.int32)
    # No host read, scalar truth, synchronization or assertions between the
    # calls: command order itself must preserve each helper's selector.
    state.clear_masked_rows(scratch, second)
    state.update_masked_rows(target, tensor([[5., 6.], [float("nan"), float("nan")]]), first, add=False)
    state.clear_masked_rows(scratch, first)
    state.copy_masked_rows(integers, tensor([[11], [22]], torch.int32), second)
    state.update_masked_rows(target, tensor([[float("nan"), float("nan")], [.5, .75]]), second, sign=-1)
    state.clear_masked_rows(scratch, first)
    state.update_masked_rows(target, tensor([[float("nan"), float("nan")], [float("nan"), float("nan")]]), none, add=False)
    state.clear_masked_rows(scratch, second)
    state.copy_masked_rows(integers, tensor([[99], [100]], torch.int32), none)
    np.testing.assert_array_equal(target.cpu().numpy(), [[5., 6.], [2.5, 3.25]])
    np.testing.assert_array_equal(integers.cpu().numpy(), [[7], [22]])
    np.testing.assert_array_equal(state._reset_mask_i32.cpu().numpy(), [1, 0])
