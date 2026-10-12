# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Real pinned compilation/lowering for inventory combinations; CPU only.

These check admission, never trajectory/contact/force parity. Each family keeps
its later numerical qualification owner and original acceptance requirements.
"""

import mujoco
import pytest

from mujoco_metal.capacity import CapacityLimits
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.stateful_actuation import ActuatorModel
from mujoco_metal.stepping import validate_stepping_profile
from mujoco_metal.support_inventory import support_inventory


def _actuator_model(axes):
  transmission, dynamics, gain, bias = axes
  target = {
      "mjTRN_JOINT": 'joint="j"', "mjTRN_JOINTINPARENT": 'jointinparent="j"',
      "mjTRN_SLIDERCRANK": 'cranksite="s" slidersite="r" cranklength=".3"',
      "mjTRN_TENDON": 'tendon="t"', "mjTRN_SITE": 'site="s" refsite="r"',
      "mjTRN_BODY": 'body="b"',
  }[transmission]
  names = [name.split("_", 1)[1].lower() for name in (dynamics, gain, bias)]
  dyn, gn, bs = names
  extra = 'actdim="2" actearly="true"' if dyn == "dcmotor" else ""
  dynprm = ".01 0 0 0 0 0 0 0 1 0" if dyn == "dcmotor" else ".01 .04 .01 0 0 0 0 0 1 0"
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0"/>
    <worldbody><site name="r" pos="0 0 1"/>
      <body name="b" pos=".1 0 1"><joint name="j" axis="0 1 0"/>
        <geom size=".05" mass="1"/><site name="s"/>
      </body></worldbody>
    <tendon><fixed name="t"><joint joint="j" coef="1"/></fixed></tendon>
    <actuator><general {target} dyntype="{dyn}" gaintype="{gn}" biastype="{bs}"
      lengthrange=".5 1.5" dynprm="{dynprm}"
      gainprm=".6 1.4 2 2 .5 1.5 1 1 1.2 0"
      biasprm=".6 1.4 2 2 .5 1.5 1 1 1.2 0" {extra}/>
    </actuator></mujoco>''')


ACTUATOR_ROWS = [row for row in support_inventory()["combinations"]
                 if row["id"].startswith("ACT:")]


@pytest.mark.parametrize("row", ACTUATOR_ROWS, ids=lambda row: row["id"])
def test_every_actuator_combination_has_real_pinned_compiler_or_admission_outcome(row):
  # This fixture enables current/thermal/integral slots for a DC gain. Non-DC
  # dynamics allocate zero/one slots, so pinned mj_dcmotorSlots rejects that
  # particular parameter configuration. Other parameter configurations may
  # admit the same enum tuple; there is no family-wide rejection claim.
  axes = row["axes"]
  invalid_dc = axes[1] != "mjDYN_DCMOTOR" and axes[2] == "mjGAIN_DCMOTOR"
  if invalid_dc:
    with pytest.raises(ValueError, match="(?i)(dc.?motor|activation)"):
      _actuator_model(axes)
    return
  model = _actuator_model(axes)
  meta = ActuatorModel(model)
  assert meta.nu == 1
  assert int(model.actuator_trntype[0]) == int(getattr(mujoco.mjtTrn, axes[0]))
  assert int(model.actuator_dyntype[0]) == int(getattr(mujoco.mjtDyn, axes[1]))
  assert int(model.actuator_gaintype[0]) == int(getattr(mujoco.mjtGain, axes[2]))
  assert int(model.actuator_biastype[0]) == int(getattr(mujoco.mjtBias, axes[3]))
  validate_stepping_profile(model, profile="integrated_euler_v1")
  assert row["qualification"] == "unqualified"
  assert row["owner"] in ("007", "019")


PAIR_ROWS = [row for row in support_inventory()["combinations"]
             if row["id"].startswith("PAIR:")]


def _pair_model(axes):
  names = [name.split("_", 1)[1].lower() for name in axes]
  asset = ('<asset><mesh name="hull" vertex="-.05 -.05 -.05 .05 -.05 -.05 '
           '.05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05 '
           '.05 .05 .05 -.05 .05 .05"/>'
           '<hfield name="hill" nrow="2" ncol="2" size="1 1 .1 .05" '
           'elevation="0 .1 .2 .3"/></asset>')
  def geom(kind, name):
    data = ('mesh="hull"' if kind in ("mesh", "sdf") else
            'hfield="hill"' if kind == "hfield" else
            'size="1 1 .1"' if kind == "plane" else 'size=".05 .05 .05"')
    return f'<geom name="{name}" type="{kind}" {data}/>'
  world, moving = [], []
  for i, kind in enumerate(names):
    element = geom(kind, f"g{i}")
    if kind in ("plane", "hfield"):
      world.append(element)
    else:
      moving.append(f'<body pos="0 0 {1+i}"><freejoint/>{element}</body>')
  return mujoco.MjModel.from_xml_string(
      f'<mujoco>{asset}<option gravity="0 0 0" sdf_initpoints="4"/>'
      f'<worldbody>{"".join(world+moving)}</worldbody>'
      '<contact><pair geom1="g0" geom2="g1"/></contact></mujoco>')


@pytest.mark.parametrize("row", PAIR_ROWS, ids=lambda row: row["id"])
def test_all_geometry_pairs_have_source_dispatcher_and_real_lowering_outcomes(row):
  model = _pair_model(row["axes"])
  # Admission alone must not be reported as contact/force/trajectory success.
  limits = CapacityLimits(max_slots=128, max_rows=1024)
  if row["admission"] == "reject absent pinned dispatcher":
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    with pytest.raises(ValueError, match="heightfield-heightfield"):
      validate_stepping_profile(model, profile="integrated_scalable_v1", limits=limits)
    return
  validate_stepping_profile(model, profile="integrated_scalable_v1", limits=limits)
  descriptor = lower_coupled_constraints(model, limits=limits)
  if row["admission"] == "empty absent pinned dispatcher":
    assert descriptor.ncontacts_max == 0
  else:
    assert descriptor.npairs == 1
  assert row["qualification"] == "unqualified"


STEP_ROWS = [row for row in support_inventory()["combinations"]
             if row["id"].startswith("STEP:")]


@pytest.mark.parametrize("row", STEP_ROWS, ids=lambda row: row["id"])
def test_integrator_solver_cone_jacobian_combinations_use_real_profile_validation(row):
  integ, solver, cone, jac = (name.split("_", 1)[1].lower() for name in row["axes"])
  xml_integrator = {"euler": "Euler", "rk4": "RK4",
                    "implicit": "implicit", "implicitfast": "implicitfast"}[integ]
  xml_solver = {"cg": "CG", "pgs": "PGS", "newton": "Newton"}[solver]
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{xml_integrator}" solver="{xml_solver}" cone="{cone}" jacobian="{jac}"/>
    <worldbody><geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .11"><freejoint/><geom size=".12" mass="1"/></body>
    </worldbody></mujoco>''')
  profile = {"euler": "integrated_scalable_v1", "rk4": "integrated_rk4_v1",
             "implicit": "integrated_implicit_v1", "implicitfast": "integrated_implicitfast_v1"}[integ]
  validate_stepping_profile(model, profile=profile)
  descriptor = lower_coupled_constraints(model)
  assert descriptor.solver_type == int(model.opt.solver)
  assert descriptor.cone_type == int(model.opt.cone)
  assert descriptor.npairs == 1
  assert row["qualification"] == "unqualified"
