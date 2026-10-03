# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R05 repair: contact manifolds, identity, and missing geometry pairs.

R05-0 codifies the pinned pair-dispatch classification (NULL entries are
intentional omissions, not gaps). Later commits add multi-contact mesh
manifolds with identity, missing valid pairs, margins/settings, and demos.
"""

import mujoco
import numpy as np
import pytest


def test_pinned_omits_plane_hfield_and_hfield_hfield_cpu():
  # mjCOLLISIONFUNC has NULL for plane-heightfield and
  # heightfield-heightfield: pinned emits zero contacts and reports
  # distmax. Native rejection/zero-slots match the engine (R05-0).
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.2 0.1"/></asset>'
      '<worldbody><geom name="pp" type="plane" size="5 5 0.1"/>'
      '<geom name="hf" type="hfield" hfield="h" pos="0 0 -0.05"/>'
      '</worldbody></mujoco>')
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  assert d.ncon == 0
  ft = np.zeros(6)
  assert mujoco.mj_geomDistance(m, d, 0, 1, 5.0, ft) == pytest.approx(5.0)
  m2 = mujoco.MjModel.from_xml_string(
      '<mujoco><asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.2 0.1"/></asset>'
      '<worldbody>'
      '<body pos="0 0 0.3"><freejoint/><geom name="a" type="hfield" hfield="h"/></body>'
      '<body pos="0 0 0.35"><freejoint/><geom name="b" type="hfield" hfield="h"/></body>'
      '</worldbody></mujoco>')
  d2 = mujoco.MjData(m2)
  for _ in range(100):
    mujoco.mj_step(m2, d2)
  assert d2.ncon == 0
