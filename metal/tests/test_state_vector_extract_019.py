# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned state-vector ordering, all signatures, and real device extraction."""
import os

import mujoco
import numpy as np
import pytest


def _model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <size nuserdata="3"/>
    <worldbody><body><joint name="a"/><geom size=".1" mass="1"/></body>
      <body pos="1 0 0"><joint name="b"/><geom size=".1" mass="1"/></body>
      <body mocap="true" pos="0 0 2"><geom size=".05"/></body></worldbody>
    <equality><joint joint1="a" joint2="b"/></equality>
    <actuator><general joint="a" dyntype="filter" dynprm=".1"/></actuator>
  </mujoco>''')


def test_state_sizes_all_pinned_masks_and_extract_individual_components():
  torch = pytest.importorskip("torch")
  from mujoco_metal.native_api import mj_stateSize, mj_extractState
  model = _model()
  for signature in range(1 << int(mujoco.mjtState.mjNSTATE)):
    assert mj_stateSize(model, signature) == mujoco.mj_stateSize(model, signature)
  for name, signature in mujoco.mjtState.__members__.items():
    assert mj_stateSize(model, signature) == mujoco.mj_stateSize(model, signature)
  full = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  source = np.arange(2*mujoco.mj_stateSize(model, full), dtype=np.float32).reshape(2, -1)
  source[0, 0] = np.nan  # Extraction is a copy utility, not a state-admission check.
  tensor = torch.tensor(source)
  for signature in [0, full, *[1 << index for index in range(14)],
                    int(mujoco.mjtState.mjSTATE_PHYSICS),
                    int(mujoco.mjtState.mjSTATE_USER)]:
    expected = np.empty((2, mujoco.mj_stateSize(model, signature)))
    for world in range(2):
      mujoco.mj_extractState(model, source[world].astype(np.float64), full,
                             expected[world], signature)
    actual = mj_extractState(model, tensor, full, signature)
    np.testing.assert_array_equal(actual.numpy(), expected.astype(np.float32))
    if actual.numel():
      actual.fill_(99)
    np.testing.assert_array_equal(tensor.numpy(), source)
  for invalid in (-1, 1 << 14):
    with pytest.raises(ValueError):
      mj_stateSize(model, invalid)
  for invalid in (True, 1.5, "1"):
    with pytest.raises(TypeError):
      mj_stateSize(model, invalid)
  with pytest.raises(ValueError, match="subset"):
    mj_extractState(model, tensor, 1, 2)
  with pytest.raises(ValueError, match="contiguous"):
    mj_extractState(model, tensor[:, :-1], full, 1)
  with pytest.raises(ValueError):
    mj_extractState(model, torch.ones((2, tensor.shape[1]), dtype=torch.int32), full, 1)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native state extraction")
def test_native_state_extraction_all_fields_owns_output_and_avoids_cpu_queries(monkeypatch):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("MPS unavailable")
  from mujoco_metal.native_api import mj_stateSize, mj_extractState
  model = _model()
  full = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  source = torch.arange(3*mj_stateSize(model, full), device="mps", dtype=torch.float32).reshape(3, -1)
  def forbidden(*args, **kwargs):
    raise AssertionError("CPU state utility invoked during device extraction")
  monkeypatch.setattr(mujoco, "mj_extractState", forbidden)
  monkeypatch.setattr(mujoco, "mj_stateSize", forbidden)
  before = source.clone()
  for destination in (0, full, *[1 << index for index in range(14)]):
    output = mj_extractState(model, source, full, destination)
    assert output.device.type == "mps"
    assert output.shape == (3, mj_stateSize(model, destination))
    output.fill_(99)
    assert torch.equal(source, before)
