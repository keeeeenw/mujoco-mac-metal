# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned plane-flex midpoint operation-order and word transport checks."""

import mujoco
import numpy as np
import pytest


def _plane_flex_model(order):
  if order == "direct":
    dof, count = "", "2 2 2"
  elif order == "trilinear":
    dof, count = ' dof="trilinear"', "2 2 2"
  else:
    dof, count = ' dof="quadratic"', "3 3 3"
  xml = f"""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom type="plane" size="0 0 .1"/>
      <flexcomp name="f" type="grid" count="{count}"
                pos=".123456789 .023456789 -.05"
                spacing=".071234567 .083456789 .02"
                mass="1" dim="3"{dof}>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.parametrize("order", ["direct", "trilinear", "quadratic"])
def test_pinned_plane_flex_midpoint_roundtrip_high_low_tail(order):
  model = _plane_flex_model(order)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == model.nflexvert
  for contact in data.contact[:data.ncon]:
    geom = int(contact.geom[0])
    flex = int(contact.flex[1])
    vertex = int(contact.vert[1])
    assert geom == 0 and flex == 0
    v = np.asarray(data.flexvert_xpos, np.float64).reshape(-1, 3)[vertex]
    pos = np.asarray(data.geom_xpos[geom], np.float64)
    matrix = np.asarray(data.geom_xmat[geom], np.float64)
    normal = matrix[[2, 5, 8]]
    radius = float(model.flex_radius[flex])
    dist = float(np.dot(v - pos, normal) - radius)
    scale = -dist * 0.5 - radius
    source_midpoint = v + normal * scale
    np.testing.assert_array_equal(source_midpoint, np.asarray(contact.pos))
    high = source_midpoint.astype(np.float32)
    low = (source_midpoint - high.astype(np.float64)).astype(np.float32)
    tail = (source_midpoint - high.astype(np.float64)
            - low.astype(np.float64)).astype(np.float32)
    represented = (high.astype(np.float64) + low.astype(np.float64)
                   + tail.astype(np.float64))
    np.testing.assert_array_equal(represented, source_midpoint)
