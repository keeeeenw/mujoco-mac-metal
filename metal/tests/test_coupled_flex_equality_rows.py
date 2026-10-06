"""Coupled row lowering follows the compiled flex-equality row inventory."""

import numpy as np
import mujoco
import pytest

pytest.importorskip("torch")

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.flex import lower_flex_descriptor


def _compiled_flex_model():
  return mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" jacobian="dense"/>
      <worldbody><body name="root"><freejoint/>
        <geom type="sphere" size=".02" mass="1"/>
        <flexcomp name="cloth" type="grid" count="2 2 1"
                  spacing=".1 .1 .1" mass="1" radius=".01" dim="2">
          <edge equality="strain"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </body></worldbody>
      <equality><flexvert flex="cloth"/></equality>
    </mujoco>
  """)


def test_coupled_flex_equality_rows_match_compiled_and_pinned_order():
  model = _compiled_flex_model()
  flex_rows = lower_flex_descriptor(model)
  coupled = lower_coupled_constraints(model)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  cursor = 0
  for eqid, eq_type in enumerate(np.asarray(model.eq_type, dtype=np.int32)):
    if eq_type not in (
        int(mujoco.mjtEq.mjEQ_FLEX),
        int(mujoco.mjtEq.mjEQ_FLEXVERT),
        int(mujoco.mjtEq.mjEQ_FLEXSTRAIN)):
      continue
    pinned_rows = np.flatnonzero(
        (np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        & (np.asarray(data.efc_id) == eqid))
    expected_ids = flex_rows.equality_row_ids_by_eqid[eqid]
    assert int(coupled.eq_rowadr[eqid]) == cursor
    assert int(coupled.eq_rownum[eqid]) == len(expected_ids) == len(pinned_rows)
    np.testing.assert_array_equal(coupled.eq_row_ids[eqid], expected_ids)
    cursor += len(expected_ids)
  assert cursor == sum(map(int, coupled.eq_rownum))
  assert int(coupled.nr_joint) >= cursor
