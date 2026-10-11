# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual public model-only contact-bound utility versus pinned 3.10.0.

Geometry tags are deliberately varied on a compiled model: mj_maxContact
reads metadata only, so no assets, MjData or collision oracle are involved.
This tests the allocation query, not a physics collider or manifold.
"""
import itertools

import mujoco
import numpy as np
import pytest

from mujoco_metal.native_api import mj_maxContact


def _model(pair=False):
  contact = '<contact><pair geom1="a" geom2="b" margin="0"/></contact>' if pair else ''
  return mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom name="a" type="sphere" size=".1"/>'
      '<geom name="b" type="sphere" size=".1" pos="1 0 0"/>'
      f'</worldbody>{contact}</mujoco>')


@pytest.mark.parametrize("types", list(itertools.product(range(9), repeat=2)))
@pytest.mark.parametrize("multiccd,nativeccd", list(itertools.product((False, True), repeat=2)))
def test_every_pinned_pair_order_and_ccd_flag(types, multiccd, nativeccd):
  model = _model()
  model.geom_type[:] = types
  model.opt.disableflags = (
      (0 if multiccd else int(mujoco.mjtDisableBit.mjDSBL_MULTICCD))
      | (0 if nativeccd else int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)))
  for seeds in (1, 40, 64):
    model.opt.sdf_initpoints = seeds
    for margin in (-2, -1, 0, 1, 2):
      expected = mujoco.mj_maxContact(model, 0, 1, margin)
      assert mj_maxContact(model, 0, 1, margin) == expected
      assert mj_maxContact(model, 1, 0, margin) == expected


@pytest.mark.parametrize("pair", (False, True))
@pytest.mark.parametrize("override", (False, True))
@pytest.mark.parametrize("geom_margin,pair_margin,override_margin",
                         list(itertools.product((0., .03), repeat=3)))
@pytest.mark.parametrize("types", ((7, 7), (6, 7), (7, 6)))
def test_model_margin_precedence_and_explicit_override(
    pair, override, geom_margin, pair_margin, override_margin, types):
  model = _model(pair)
  model.geom_type[:] = types
  model.geom_margin[:] = [geom_margin, 0.]
  model.pair_margin[:] = pair_margin
  model.opt.o_margin = override_margin
  model.opt.enableflags = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE) if override else 0
  before = (model.geom_type.copy(), model.geom_margin.copy(),
            model.pair_margin.copy(), int(model.opt.enableflags), float(model.opt.o_margin))
  for ids in ((0, 1), (1, 0)):
    for margin in (-2, -1, 0, 1, 2):
      assert mj_maxContact(model, *ids, margin) == mujoco.mj_maxContact(model, *ids, margin)
  np.testing.assert_array_equal(model.geom_type, before[0])
  np.testing.assert_array_equal(model.geom_margin, before[1])
  np.testing.assert_array_equal(model.pair_margin, before[2])
  assert (int(model.opt.enableflags), float(model.opt.o_margin)) == before[3:]


def test_metadata_query_does_not_dispatch_upstream_or_device_physics(monkeypatch):
  model = _model()
  expected = mujoco.mj_maxContact(model, 0, 1, -1)
  def forbidden(*args, **kwargs):
    raise AssertionError("metadata query dispatched physics or upstream utility")
  for name in ("MjData", "mj_maxContact", "mj_forward", "mj_step", "mj_collision"):
    monkeypatch.setattr(mujoco, name, forbidden)
  assert mj_maxContact(model, np.int32(0), np.int64(1)) == expected


@pytest.mark.parametrize("bad", (True, 0., "0", None, 1 << 31, -(1 << 31) - 1))
@pytest.mark.parametrize("position", (0, 1, 2))
def test_invalid_argument_domains_are_rejected_before_reads(bad, position):
  args = [0, 1, -1]
  args[position] = bad
  with pytest.raises(ValueError):
    mj_maxContact(_model(), *args)


def test_invalid_ids_and_wrong_model_are_rejected():
  model = _model()
  for ids in ((-1, 0), (0, -1), (model.ngeom, 0), (0, model.ngeom)):
    with pytest.raises(ValueError, match="geometry IDs"):
      mj_maxContact(model, *ids)
  with pytest.raises(TypeError, match="MjModel"):
    mj_maxContact(object(), 0, 1)


def test_full_signed_int32_margin_domain_and_same_geom_queries():
  model = _model()
  model.geom_type[:] = 7
  for margin in (-(1 << 31), (1 << 31) - 1):
    for ids in ((0, 0), (1, 1), (0, 1)):
      assert mj_maxContact(model, *ids, margin) == mujoco.mj_maxContact(model, *ids, margin)
