# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned CPU preflight inventory for the still-gated flex 018 routes.

These tests exercise MuJoCo 3.10's actual compiled model and contact/equality
records against the public host lowerers. They do not qualify the native
narrowphase or open production admission.
"""

import numpy as np
import mujoco
import pytest

from mujoco_metal.flex_contact import (
    _KIND_ELEMENT_PAIR,
    _KIND_GEOM_ELEMENT,
    _KIND_PLANE_VERTEX,
    lower_flex_contacts,
)


def _xml_flex(name, dim, count, pos="0 0 0", dof=None,
              selfcollide="none", shell=False, contype=1, conaffinity=1):
  dof_attr = "" if dof is None else f' dof="{dof}"'
  elasticity = (' young="100" poisson=".2" thickness=".01"'
                ' elastic2d="stretch"' if shell else
                ' young="100" poisson=".2"')
  return f"""<flexcomp name="{name}" type="grid" count="{count}"
      pos="{pos}" spacing=".05 .05 .05" mass="1" dim="{dim}"{dof_attr}>
    <contact contype="{contype}" conaffinity="{conaffinity}"
             selfcollide="{selfcollide}" condim="1"/>
    <elasticity{elasticity}/>
  </flexcomp>"""


def _cpu_data(model, deform=None):
  """Run the pinned oracle from the same float32 generalized state as MPS."""
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(data.qpos, np.float32).astype(np.float64)
  data.qvel[:] = np.asarray(data.qvel, np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  if deform is not None:
    rest = np.asarray(data.flexvert_xpos, np.float64).reshape(-1, 3).copy()
    for vertex, body in enumerate(np.asarray(model.flex_vertbodyid, np.int32)):
      joint = int(model.body_jntadr[int(body)])
      qadr = int(model.jnt_qposadr[joint])
      target = deform(rest[vertex], vertex)
      data.qpos[qadr:qadr+3] = np.asarray(target-rest[vertex], np.float32).astype(np.float64)
    data.qpos[:] = np.asarray(data.qpos, np.float32).astype(np.float64)
    mujoco.mj_forward(model, data)
  return data


def _assert_contacts_have_fixed_candidates(model, data, descriptor, *, self_contact):
  """Every pinned CPU element pair has a stable lowered ordinal span."""
  pair_contacts = {}
  for contact in data.contact[:data.ncon]:
    if not np.all(np.asarray(contact.elem) >= 0):
      continue
    f1, f2 = map(int, contact.flex)
    e1, e2 = map(int, contact.elem)
    if f1 < 0 or f2 < 0:
      continue
    if self_contact:
      assert f1 == f2
      key = (f1, min(e1, e2), f2, max(e1, e2))
    else:
      key = (f1, e1, f2, e2)
    pair_contacts.setdefault(key, []).append(contact)
  assert pair_contacts, "fixture must produce real pinned flex/flex contacts"

  for (f1, e1, f2, e2), contacts in pair_contacts.items():
    candidates = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == f1) & (descriptor.elem1 == e1)
        & (descriptor.flex2 == f2) & (descriptor.elem2 == e2))
    candidates = candidates[np.argsort(descriptor.contact_ordinal[candidates])]
    assert candidates.size >= len(contacts), (
        f"compiled slot inventory under-reserves pinned pair {(f1,e1,f2,e2)}: "
        f"{candidates.size} slots for {len(contacts)} contacts")
    assert np.all(descriptor.row_span[candidates] > 0)
    assert np.all(descriptor.row_start[candidates[1:]] >=
                  descriptor.row_start[candidates[:-1]] +
                  descriptor.row_span[candidates[:-1]])


@pytest.mark.parametrize(("dim", "count"), [(2, "3 3 1"), (3, "2 2 2")])
def test_cross_flex_direct_simplex_cpu_contacts_fit_public_fixed_slots(dim, count):
  """Triangle/tetra element pairs from pinned CCD map to distinct slots."""
  offset = ".03 0 0" if dim == 2 else ".03 .01 .01"
  flexes = (_xml_flex("a", dim, count, "0 0 0", contype=1, conaffinity=2)
            + _xml_flex("b", dim, count, offset,
                        contype=2, conaffinity=1))
  model = mujoco.MjModel.from_xml_string(
      f"<mujoco><option gravity='0 0 0'/><worldbody>{flexes}</worldbody></mujoco>")
  data = _cpu_data(model)
  descriptor = lower_flex_contacts(model)
  assert np.all(model.flex_dim == dim)
  _assert_contacts_have_fixed_candidates(model, data, descriptor,
                                         self_contact=False)
  assert descriptor.row_capacity == int(np.sum(descriptor.row_span))


@pytest.mark.parametrize(("dim", "count"), [(2, "4 4 1"), (3, "3 3 3")])
def test_self_flex_direct_simplex_cpu_contacts_fit_public_fixed_slots(dim, count):
  """Folded 2D triangles and compressed 3D tetrahedra retain pair slots."""
  flex_xml = _xml_flex("self", dim, count, selfcollide="narrow")
  model = mujoco.MjModel.from_xml_string(
      f"<mujoco><option gravity='0 0 0'/><worldbody>{flex_xml}</worldbody></mujoco>")
  if dim == 2:
    def deform(point, _vertex):
      out = point.copy()
      if point[1] > .051:
        out[1] -= .1
      return out
  else:
    def deform(point, _vertex):
      # Collapse separated tetrahedral cells into overlapping, non-adjacent
      # features while retaining distinct compiled bodies.
      return point * .15
  data = _cpu_data(model, deform)
  descriptor = lower_flex_contacts(model)
  assert int(model.flex_selfcollide[0]) != int(mujoco.mjtFlexSelf.mjFLEXSELF_NONE)
  _assert_contacts_have_fixed_candidates(model, data, descriptor,
                                         self_contact=True)


@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_interpolated_flex_cpu_surface_contact_and_compiled_material_inventory(
    dof, shell):
  """Q1/Q2 and shell interpolation stay distinguishable in host lowering."""
  plane = '<geom name="floor" type="plane" pos="0 0 0" size="0 0 .1" contype="0" conaffinity="1" condim="1"/>'
  flex = _xml_flex("interp", 3, "3 3 3", "0 0 -.02", dof=dof,
                   shell=shell, contype=1, conaffinity=0)
  model = mujoco.MjModel.from_xml_string(
      f"<mujoco><option gravity='0 0 0'/><worldbody>{plane}{flex}</worldbody></mujoco>")
  data = _cpu_data(model)
  descriptor = lower_flex_contacts(model)
  from mujoco_metal.flex import lower_flex_descriptor
  material = lower_flex_descriptor(model)
  order = 1 if dof == "trilinear" else 2
  assert int(model.flex_interp[0]) == (-order if shell else order)
  assert material is not None and int(material.interp[0]) == int(model.flex_interp[0])
  assert int(material.stiffnessadr[0]) == int(model.flex_stiffnessadr[0])
  assert material.stiffness.size == np.asarray(model.flex_stiffness).size
  if not shell:
    # Pin a nonzero constitutive CPU witness as well as the compiled map. The
    # signed shell cases are inventories only here; they require separate
    # shell-force/tangent qualification.
    strained = mujoco.MjData(model)
    strained.qpos[:] = np.asarray(model.qpos0, np.float32).astype(np.float64)
    strained.qpos[-1] += .002
    strained.qpos[:] = np.asarray(strained.qpos, np.float32).astype(np.float64)
    mujoco.mj_forward(model, strained)
    assert np.linalg.norm(strained.qfrc_spring) > 1e-5
  pinned = [c for c in data.contact[:data.ncon]
            if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
            and int(c.vert[1]) >= 0]
  assert pinned, "plane fixture must produce pinned interpolated vertex contacts"
  for contact in pinned:
    vertex = int(contact.vert[1]) + int(model.flex_vertadr[0])
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_PLANE_VERTEX)
        & (descriptor.geom == 0) & (descriptor.flex1 == 0)
        & (descriptor.vert1 == vertex))
    assert slots.size == 1
    assert int(descriptor.row_span[slots[0]]) == 1
  # flex_vertbodyid is deliberately not a usable rigid owner for Q1/Q2
  # vertices; the compiled interpolation support lives in flex_nodebodyid.
  assert np.all(np.asarray(model.flex_vertbodyid)[
      int(model.flex_vertadr[0]):int(model.flex_vertadr[0])+int(model.flex_vertnum[0])] < 0)
  assert int(model.flex_nodenum[0]) > 0
  assert np.asarray(model.flex_nodebodyid).size >= int(model.flex_nodenum[0])


def test_flex_equality_rows_and_integrator_preflight_use_compiled_inventory():
  """Equality row counts follow pinned efc ids; unsupported RK4 stays rejected."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" timestep=".001" integrator="Euler"/>
      <worldbody><flexcomp name="cloth" type="grid" count="2 2 1"
          spacing=".1 .1 .1" mass="1" radius=".01" dim="2">
        <contact contype="0" conaffinity="0"/>
        <edge equality="strain"/>
      </flexcomp></worldbody>
      <equality><flexvert flex="cloth"/></equality>
    </mujoco>
  """)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  from mujoco_metal.flex import lower_flex_descriptor
  descriptor = lower_flex_descriptor(model)
  assert descriptor is not None
  equality_types = np.asarray(model.eq_type, np.int32)
  vert_ids = np.flatnonzero(equality_types == int(mujoco.mjtEq.mjEQ_FLEXVERT))
  assert vert_ids.size == 1
  eqid = int(vert_ids[0])
  actual_rows = np.flatnonzero(
      (np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
      & (np.asarray(data.efc_id) == eqid))
  assert descriptor.equality_row_counts[eqid] == actual_rows.size
  np.testing.assert_array_equal(descriptor.equality_row_ids_by_eqid[eqid],
                                np.arange(actual_rows.size, dtype=np.int32))
  from mujoco_metal.stepping import validate_stepping_profile
  validate_stepping_profile(model, profile="integrated_scalable_v1")
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
  with pytest.raises(ValueError, match="Euler integrator"):
    validate_stepping_profile(model, profile="integrated_scalable_v1")


def test_articulated_flex_attachment_has_pinned_offcenter_passive_force():
  """CPU preflight keeps the hinge parent and generated flex bodies distinct."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="arm" pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="attached" type="grid" count="2 2 1"
                  pos=".13 .21 -.07" spacing=".1 .1 .1" mass="1" dim="2">
          <contact contype="0" conaffinity="0"/>
          <edge stiffness="0" damping="0"/>
          <elasticity young="1000" poisson=".2" damping=".4"
                      thickness=".01" elastic2d="stretch"/>
        </flexcomp>
      </body>
    </worldbody></mujoco>
  """)
  assert int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_HINGE)
  qpos = np.asarray(model.qpos0, np.float64).copy()
  qpos[0] = .31
  qpos[1] += .035
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = np.linspace(-.1, .1, model.nv)
  mujoco.mj_forward(model, data)
  attached_bodies = np.asarray(model.flex_vertbodyid, np.int32)
  assert np.all(attached_bodies > 0)
  assert np.linalg.norm(data.qfrc_passive) > 1e-3
  from mujoco_metal.flex import lower_flex_descriptor
  descriptor = lower_flex_descriptor(model)
  assert descriptor is not None
  assert int(descriptor.njnt) == model.njnt
  assert descriptor.vertbodyid.size == model.nflexvert


def test_sleep_profile_and_cross_flex_link_inventory_are_static_and_fixed():
  """Sleep preflight has stable cross-tree links for real flex candidates."""
  flexes = (_xml_flex("a", 3, "2 2 2", "0 0 0", contype=1, conaffinity=2)
            + _xml_flex("b", 3, "2 2 2", ".03 .01 .01",
                        contype=2, conaffinity=1))
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option gravity='0 0 0' timestep='.001'>"
      "<flag sleep='enable'/></option><worldbody>"
      + flexes + "</worldbody></mujoco>")
  descriptor = lower_flex_contacts(model)
  from mujoco_metal.stepping import validate_stepping_profile
  plan = validate_stepping_profile(model, profile="integrated_scalable_v1")
  assert plan is not None and model.ntree > 1
  links = np.asarray(descriptor.link_tree_pairs, np.int32)
  candidate_ids = np.asarray(descriptor.candidate_link_ids, np.int32)
  assert links.ndim == 2 and links.shape[1] == 2
  assert links.shape[0] == descriptor.link_capacity > 0
  assert np.all(links[:, 0] < links[:, 1])
  assert len({tuple(map(int, pair)) for pair in links}) == links.shape[0]
  assert candidate_ids.shape[0] == descriptor.slot_count
  assert np.all((candidate_ids < 0) | (candidate_ids < descriptor.link_capacity))


def test_nonplane_production_path_remains_explicitly_guarded():
  """A compiled convex flex contact is rejected before CPU/GPU row work."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.flex_contact import FlexContactProgram
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom name="sphere" type="sphere" size=".2"
            contype="0" conaffinity="1"/>
      <flexcomp name="volume" type="grid" count="2 2 2"
          spacing=".1 .1 .1" mass="1" dim="3">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <elasticity young="100" poisson=".2"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  program = FlexContactProgram(model, device="cpu")
  assert program.descriptor.global_enabled
  assert not program._narrowphase_admitted
  with pytest.raises(NotImplementedError, match="support/CCD/manifold"):
    program.run_device(
        torch.zeros((1, model.nflexvert, 3)),
        torch.zeros((1, model.ngeom, 3)),
        torch.tensor([[[1., 0., 0., 0.]]]))


def test_original_scale_q2_volume_solver_preflight_keeps_full_capacity():
  """Host lowering preserves the 87-DOF/359-row Q2 sphere fixture budget."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".001" gravity="0 0 0" cone="elliptic"
                   solver="Newton" iterations="80" tolerance="1e-9"
                   jacobian="sparse"/>
      <worldbody>
        <body name="ball" pos=".1 .1 .1"><freejoint/>
          <geom name="sphere" type="sphere" size=".08" condim="3"
                friction=".8 .01 .001" solref=".02 1"
                solimp=".9 .95 .001"/>
        </body>
        <flexcomp name="interp" type="grid" count="3 3 3"
                  pos="0 0 .1" spacing=".1 .1 .1" mass="1" dim="3"
                  dof="quadratic">
          <contact contype="1" conaffinity="0" selfcollide="none"
                   condim="3" friction=".8 .01 .001" solref=".02 1"
                   solimp=".9 .95 .001"/>
          <elasticity young="100" poisson=".2"/>
        </flexcomp>
      </worldbody>
    </mujoco>
  """)
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  data = _cpu_data(model)
  contacts = [contact for contact in data.contact[:data.ncon]
              if int(contact.flex[1]) == 0 and int(contact.elem[1]) >= 0]
  assert len(contacts) == 8
  slots = lower_flex_contacts(model)
  coupled = lower_coupled_constraints(
      model, limits=CapacityLimits(max_nv=128, max_slots=512,
                                   max_pairs=512, max_rows=512))
  assert (model.nq, model.nv) == (88, 87)
  assert slots.slot_count == 36 and slots.row_capacity == 108
  assert coupled.n_flex_contact_rows == 108
  assert coupled.nr_joint == 251 and coupled.nr == 359
  assert int(model.opt.iterations) == 80
  assert int(model.opt.solver) == int(mujoco.mjtSolver.mjSOL_NEWTON)
