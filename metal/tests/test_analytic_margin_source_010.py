# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compiled collision-margin operands and their actual allocation contract."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import _runtime_buffer_sizes
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from test_analytic_material_lifecycle_010 import _model, _state


def _many_cylinders():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" o_margin=".006"><flag override="enable"/></option>
    <worldbody>
      <body pos="0 0 .5"><freejoint/><geom type="cylinder" size=".0625 .125"/></body>
      <body pos=".1240234375 0 .5"><freejoint/><geom type="cylinder" size=".0625 .125"/></body>
      <body pos=".5 0 .5"><freejoint/><geom type="cylinder" size=".0625 .125"/></body>
    </worldbody></mujoco>''')


def test_source_margin_rounding_changes_pinned_restart_frame():
  normals = []
  for rounded in (False, True):
    model, half = _model("cylinder", "cylinder", False, "pyramidal", "override")
    q, v = _state(model, "cylinder", half, moving=True)
    if rounded:
      model.opt.o_margin = float(np.float32(model.opt.o_margin))
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mujoco.mj_forward(model, data)
    assert data.ncon == 4
    normals.append(np.asarray([contact.frame[:3].copy() for contact in data.contact]))
  # At identical poses, rounding only the compiled margin flips the
  # deterministic perturbed-frame tilt. This is not contact-order noise.
  assert np.max(np.abs(normals[0] - normals[1])) > .0019


def test_source_margin_words_and_planar_constants_have_budgeted_backings():
  model = _many_cylinders()
  descriptor = lower_coupled_constraints(model)
  assert descriptor.npairs == 3
  words = descriptor.pair_margin_source_words.reshape(3, -1)
  np.testing.assert_array_equal(words.astype(float).sum(axis=0), model.opt.o_margin)
  sizes = dict(_runtime_buffer_sizes(model, 2, npairs=descriptor.npairs))
  assert sizes["model.coupled.pair_margin_source"] == words.size == 9
  # Constructor preflight has not selected a coupled topology yet.
  assert dict(_runtime_buffer_sizes(model, 2))["model.coupled.pair_margin_source"] == 3
  planar = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .09"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>''')
  sizes = dict(_runtime_buffer_sizes(planar, 1, npairs=1))
  assert sizes["model.coupled.geom_size_source"] == 9 * planar.ngeom


@pytest.mark.parametrize("first,second", [("capsule", "box"), ("cylinder", "box"),
                                         ("ellipsoid", "box")])
def test_callback_margin_includes_gap_but_solver_margin_does_not(first, second):
  model, _ = _model(first, second, False, "pyramidal", "explicit",
                    margin=.015625, gap=.0078125)
  descriptor = lower_coupled_constraints(model)
  words = descriptor.pair_margin_source_words.reshape(3, -1).astype(float)
  np.testing.assert_array_equal(words.sum(axis=0), model.pair_margin + model.pair_gap)
  np.testing.assert_array_equal(descriptor.margin, model.pair_margin)
  np.testing.assert_array_equal(descriptor.gap, model.pair_gap)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual source-margin backing")
def test_actual_three_pair_margin_backing_and_executed_contacts(monkeypatch):
  from mujoco_metal import MetalSimulation
  model = _many_cylinders()
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  program = sim._coupled_constraints
  source = program._constants["pair_margin_source"]
  assert source.numel() == 9
  np.testing.assert_array_equal(source.cpu().numpy().reshape(3, -1).astype(float).sum(0),
                                model.opt.o_margin)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  with monkeypatch.context() as context:
    def forbidden(*args, **kwargs):
      raise AssertionError("source-margin collision used CPU physics")
    for name in ("mj_step", "mj_forward", "mj_collision", "mj_kinematics"):
      context.setattr(mujoco, name, forbidden)
    assembly = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(assembly["status"].cpu().numpy(), [0, 0])
  assert data.ncon > 0 and assembly["contact_mask"].sum().item() == 2 * data.ncon
  for world in range(2):
    np.testing.assert_allclose(assembly["qfrc_constraint"][world].cpu().numpy(),
                              data.qfrc_constraint, rtol=.03, atol=.02)
