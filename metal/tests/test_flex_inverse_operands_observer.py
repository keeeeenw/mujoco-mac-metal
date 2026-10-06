"""Opt-in observation-only test for the dense flex equality inverse stage."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mujoco
import numpy as np
import pytest

from test_public_enable_history_plugin_composition_019 import (
    _flex_equality_model, _host)


def _copy(value):
  if value is None:
    return None
  if hasattr(value, "detach"):
    return value.detach().cpu().numpy().copy()
  if isinstance(value, np.ndarray):
    return value.copy()
  return value


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native test")
def test_observe_flex_inverse_operands_dense():
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_inverse, mj_inverseSkip, mj_fwdPosition
  from mujoco_metal import inverse_constraints
  from test_public_enable_history_plugin_composition_019 import _set_jacobian_layout_for_profile

  profile = "integrated_euler_v1"
  model = _flex_equality_model()
  _set_jacobian_layout_for_profile(model, profile)
  qpos = model.qpos0.astype(np.float32).copy()
  qpos[0] += .01
  qvel = np.linspace(-.2, .3, model.nv, dtype=np.float32)
  sim = MetalSimulation(model, batch_size=1, qpos=qpos[None, :],
                        qvel=qvel[None, :], profile=profile)
  ref = mujoco.MjData(model)
  ref.qpos[:] = qpos
  ref.qvel[:] = qvel
  for _ in range(3):
    status = sim.step()
    assert not status.detach().cpu().numpy().any()
    mujoco.mj_step(model, ref)
  native_qpos = _host(sim.state._qpos)[0]
  native_qvel = _host(sim.state._qvel)[0]
  target = ref.qacc.astype(np.float32).astype(np.float64)
  qacc = torch.as_tensor(target[None, :], dtype=torch.float32, device="mps")

  saved = {}
  original_force = inverse_constraints.inverse_constraint_force
  original_skip = sim.inverse_skip

  def force_observer(rows, acceleration, descriptor, **kwargs):
    for key in ("R", "ar", "ar_low", "lo", "hi", "active", "rhs",
                "row_impedance", "J"):
      if key in rows:
        saved["native_" + key] = _copy(rows[key])
    saved["native_qacc_used"] = _copy(acceleration)
    saved["native_eq_rows"] = int(descriptor.n_eq_rows)
    saved["native_nr"] = int(descriptor.nr)
    saved["native_nv"] = int(descriptor.nv)
    result = original_force(rows, acceleration, descriptor, **kwargs)
    if isinstance(result, tuple):
      saved["native_constraint_force"] = _copy(result[0])
      saved["native_constraint_force_low"] = _copy(result[1])
    else:
      saved["native_constraint_force"] = _copy(result)
    return result

  def skip_observer(*args, **kwargs):
    kwargs["return_components"] = True
    result = original_skip(*args, **kwargs)
    for name, value in result.get("qfrc_inverse_components", {}).items():
      saved["native_component_" + name] = _copy(value)
    record = result.get("record")
    if record is not None:
      saved["native_context"] = _copy(
          sim._coupled_constraints._workspace["position_assembly_context"])
      from mujoco_metal.forward_stages import ForwardStage
      saved["native_velocity"] = _copy(
          record.values[ForwardStage.VEL]["qvel"])
      saved["native_stage_keys"] = np.asarray(
          [len(record.values)], dtype=np.int32)
    return result

  inverse_constraints.inverse_constraint_force = force_observer
  sim.inverse_skip = skip_observer
  try:
    native_force = _host(mj_inverse(sim, qacc=qacc))[0]
  finally:
    inverse_constraints.inverse_constraint_force = original_force
    sim.inverse_skip = original_skip

  cpu = mujoco.MjData(model)
  cpu.qpos[:] = native_qpos
  cpu.qvel[:] = native_qvel
  cpu.qacc[:] = target
  mujoco.mj_inverse(model, cpu)
  # Exercise the same cached POS -> inverse VEL path independently from the
  # full query; this is the path that consumes retained flex B/K coefficients.
  pos_record = mj_fwdPosition(sim, return_record=True, skipsensor=True)
  split_force = _host(mj_inverseSkip(
      sim, skipstage=mujoco.mjtStage.mjSTAGE_POS, record=pos_record,
      qacc=qacc))[0]
  np.testing.assert_allclose(split_force, cpu.qfrc_inverse,
                             atol=8e-3, rtol=8e-3,
                             err_msg="split POS-to-VEL flex inverse")
  saved["split_inverse_force"] = split_force.copy()
  nefc = int(cpu.nefc)
  dense_j = np.zeros((nefc, model.nv), dtype=np.float64)
  for col in range(model.nv):
    basis = np.zeros(model.nv, dtype=np.float64)
    basis[col] = 1.0
    product = np.zeros(nefc, dtype=np.float64)
    mujoco.mj_mulJacVec(model, cpu, product, basis)
    dense_j[:, col] = product
  for key, value in {
      "native_qpos": native_qpos, "native_qvel": native_qvel,
      "target_qacc": target, "native_inverse_force": native_force,
      "cpu_J": dense_j, "cpu_R": cpu.efc_R[:nefc],
      "cpu_aref": cpu.efc_aref[:nefc], "cpu_force_rows": cpu.efc_force[:nefc],
      "cpu_qfrc_constraint": cpu.qfrc_constraint,
      "cpu_qfrc_passive": cpu.qfrc_passive, "cpu_qfrc_bias": cpu.qfrc_bias,
      "cpu_inverse_force": cpu.qfrc_inverse,
  }.items():
    saved[key] = np.asarray(value).copy()
  detail = {
      "profile": profile, "native_nefc": nefc,
      "native_nr": saved.get("native_nr"),
      "native_eq_rows": saved.get("native_eq_rows"),
      "native_force": native_force.tolist(),
      "cpu_force": cpu.qfrc_inverse.tolist(),
      "max_force_error": float(np.max(np.abs(native_force - cpu.qfrc_inverse))),
      "captured": sorted(saved),
  }
  print("flex-inverse-operands " + json.dumps(detail, sort_keys=True))
  capture = os.getenv("MUJOCO_METAL_INVERSE_OPERAND_CAPTURE")
  if capture:
    np.savez(capture, **{key: value for key, value in saved.items()
                         if isinstance(value, np.ndarray)})
  np.testing.assert_allclose(native_force, cpu.qfrc_inverse, atol=8e-3, rtol=8e-3)
