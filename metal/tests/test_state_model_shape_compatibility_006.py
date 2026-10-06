"""Model-change state-shape admission for milestone 006."""

import mujoco
import numpy as np
import pytest
from types import SimpleNamespace


def _model(userdata):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option><flag contact="disable"/></option><size nuserdata="{userdata}"/><worldbody>'
      '<body><joint type="hinge"/><geom type="sphere" size=".1" mass="1"/>'
      '</body></worldbody></mujoco>')


@pytest.mark.parametrize("field", ["nhistory", "nuserdata", "npluginstate"])
def test_lifecycle_hidden_state_widths_are_checked_before_any_build(field):
  from mujoco_metal.simulation import _assert_lifecycle_state_shapes_compatible

  old = SimpleNamespace(**{field: 2})
  new = SimpleNamespace(**{field: 3})
  with pytest.raises(ValueError, match=field):
    _assert_lifecycle_state_shapes_compatible(old, new)


def test_lifecycle_matching_hidden_widths_are_compatible():
  from mujoco_metal.simulation import _assert_lifecycle_state_shapes_compatible

  old = SimpleNamespace(nhistory=7, nuserdata=3, npluginstate=5)
  new = SimpleNamespace(nhistory=7, nuserdata=3, npluginstate=5)
  _assert_lifecycle_state_shapes_compatible(old, new)


def test_lifecycle_rejects_userdata_shape_change_before_stage_swap_cpu():
  """A new per-world state width requires a rebuilt Simulation."""
  from mujoco_metal.simulation import _assert_lifecycle_state_shapes_compatible

  old = _model(0)
  new = _model(2)
  for name in ("nq", "nv", "nbody", "njnt", "ngeom", "nsite", "ntendon",
               "nu", "na", "neq", "nmocap"):
    assert getattr(old, name) == getattr(new, name)
  assert old.nuserdata == 0 and new.nuserdata == 2

  with pytest.raises(ValueError, match="nuserdata"):
    _assert_lifecycle_state_shapes_compatible(old, new)


def test_apply_lifecycle_checks_hidden_width_before_constructing_device_stages():
  from mujoco_metal.simulation import MetalSimulation

  old, new = _model(0), _model(2)
  sim = object.__new__(MetalSimulation)
  sim._mjmodel = old
  with pytest.raises(ValueError, match="nuserdata"):
    MetalSimulation.apply_lifecycle(sim, new)
  assert sim._mjmodel is old


def test_device_state_adoption_is_atomic_for_hidden_model_sized_fields_cpu():
  pytest.importorskip("torch")
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  old, new = _model(0), _model(2)
  old_state = DeviceState(old, validate_stepping_profile(old, profile="contact_free_sensor_euler_v1"), 2, device="cpu")
  new_state = DeviceState(new, validate_stepping_profile(new, profile="contact_free_sensor_euler_v1"), 2, device="cpu")
  before = (old_state._model, old_state._model_fingerprint,
            old_state._profile_fingerprint, old_state.generation,
            old_state._userdata.clone(), old_state._plugin_state.clone())
  with pytest.raises(ValueError, match="nuserdata"):
    old_state.adopt_descriptor(new_state)
  assert old_state._model is before[0]
  assert old_state._model_fingerprint == before[1]
  assert old_state._profile_fingerprint == before[2]
  assert old_state.generation == before[3]
  import torch
  torch.testing.assert_close(old_state._userdata, before[4], rtol=0, atol=0)
  torch.testing.assert_close(old_state._plugin_state, before[5], rtol=0, atol=0)


def _history_model(delay):
  spec = mujoco.MjSpec.from_string('''<mujoco>
    <option timestep=".002" gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody><actuator><motor name="motor" joint="j"/></actuator>
    <sensor><jointpos name="position" joint="j"/></sensor></mujoco>''')
  spec.actuator("motor").nsample = 8
  spec.actuator("motor").interp = 1
  spec.actuator("motor").delay = delay
  spec.sensor("position").nsample = 8
  spec.sensor("position").interp = 1
  spec.sensor("position").delay = delay
  return spec.compile()


def test_model_lifecycle_resets_compiled_history_but_preserves_live_dynamics_cpu():
  import pytest
  torch = pytest.importorskip("torch")
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  old, new = _history_model(.003), _history_model(.005)
  assert old.nhistory == new.nhistory
  old_state = DeviceState(old, validate_stepping_profile(old, profile="contact_free_sensor_euler_v1"), 2, device="cpu")
  new_state = DeviceState(new, validate_stepping_profile(new, profile="contact_free_sensor_euler_v1"), 2, device="cpu")
  old_state._qpos.fill_(.25)
  old_state._qvel.fill_(-.5)
  old_state._history.fill_(19.)
  history_storage = old_state._history

  old_state.adopt_descriptor(new_state)
  old_state.reset_model_history(new_state)
  assert old_state._history is history_storage
  torch.testing.assert_close(old_state._qpos, torch.full_like(old_state._qpos, .25),
                             rtol=0, atol=0)
  torch.testing.assert_close(old_state._qvel, torch.full_like(old_state._qvel, -.5),
                             rtol=0, atol=0)
  torch.testing.assert_close(old_state._history, new_state._history, rtol=0, atol=0)
  np.testing.assert_array_equal(old_state._history0, new_state._history0)
