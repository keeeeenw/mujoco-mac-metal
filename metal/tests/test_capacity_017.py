# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 017: explicit capacity architecture (CPU-only part).

Deterministic estimation, user limits, memory budgeting, overflow
diagnostics, and growth-at-host-boundaries. No physics changes here.
"""

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import (
    BASE_MAX_PAIRS,
    BASE_MAX_ROWS,
    BASE_MAX_SLOTS,
    BASE_NVIDIA_NV,
    AUTO_JACOBIAN_DENSE_NV,
    DENSE_ROW_THRESHOLD,
    CapacityLimits,
    CapacityOverflow,
    check_capacity,
    estimate_capacity,
    estimate_workspace,
    _check_i32_elements,
    SIGNED_I32_ELEMENT_LIMIT,
    selected_jacobian_kind,
    primal_scratch_floats,
)


def _model(nv_hint="slide-chain"):
  if nv_hint == "slide-chain":
    bodies = "".join(
        f'<body pos="{0.2 * k} 0 0.3"><joint name="s{k}" type="slide" axis="1 0 0"/>'
        f'<geom type="sphere" size="0.05"/></body>' for k in range(4))
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")
  return mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body><joint/><geom type='sphere' size='.1'/></body></worldbody></mujoco>")


def test_estimate_is_deterministic_and_counts_memory():
  model = _model()
  est1 = estimate_capacity(model, 2, npairs=3, nslots=5, nr=20,
                           magnetic_force_plugins=1,
                           site_feedback_plugins=1)
  est2 = estimate_capacity(model, 2, npairs=3, nslots=5, nr=20,
                           magnetic_force_plugins=1,
                           site_feedback_plugins=1)
  assert est1 == est2
  assert est1.dense_path == (20 <= DENSE_ROW_THRESHOLD)
  assert est1.memory_bytes == sum(v for _, v in est1.memory_breakdown)
  assert est1.memory_bytes > 0
  names = [k for k, _ in est1.memory_breakdown]
  # R08a: parts mirror prepare_workspace buffer-for-buffer.
  assert "workspace_debug" in names and "contact_row_data" in names
  assert "workspace_J" in names and "out_contact_force" in names
  assert "eq_active_default" in names
  solver_parts = estimate_workspace_expected(model, 2, 3, 5, 20)
  breakdown = dict(est1.memory_breakdown)
  assert all(breakdown[name] == size for name, size in solver_parts)
  runtime = {name: size for name, size in est1.memory_breakdown
             if name not in dict(solver_parts)}
  assert runtime["smooth.crb"] == 2 * model.nbody * 36 * 4
  assert runtime["smooth.local_inertia"] == 2 * model.nbody * 36 * 4
  assert runtime["smooth.cdof_dot"] == 2 * model.nv * 6 * 4
  assert runtime["smooth.position_cache.mass_matrix"] == (
      2 * model.nv * model.nv * 4)
  assert runtime["smooth.position_cache.poses.body_pos"] == (
      2 * model.nbody * 3 * 4)
  assert runtime["model.smooth.dof_treeid"] == max(model.nv, 1) * 4
  assert runtime["simulation.body_wrench"] == 2 * model.nbody * 6 * 4
  assert runtime["simulation.fixed_tendon.last_length"] == (
      2 * max(model.ntendon, 1) * 4)
  assert runtime["simulation.fixed_tendon.dims"] == (8 + 2) * 4
  assert runtime["simulation.fixed_tendon.velocity_dims"] == (8 + 2) * 4
  assert runtime["passive.gravcomp_output"] == max(model.nv, 1) * 2 * 4
  assert runtime["passive.gravcomp_zero_wrench"] == (
      2 * model.nbody * 6 * 4)
  assert runtime["passive.force"] == 2 * model.nv * 4
  assert runtime["passive.wrench"] == 2 * model.nbody * 6 * 4
  assert runtime["passive.springpoly"] == max(model.njnt * 2, 1) * 4
  assert runtime["passive.fk.model.body_pos"] == max(model.nbody * 3, 1) * 4
  assert runtime["model.smooth.tendon_j_colind"] == max(
      model.ten_J_colind.size, 1) * 4
  assert runtime["passive.fk.model.mass_colind"] == max(
      model.M_colind.size, 1) * 4
  # FK constants and high/low/tail outputs share physical arenas. The
  # admission envelope must count their backing allocations, not old views.
  assert runtime["fk.dims"] == 61 * 4
  int_words = (2 * max(model.nbody, 1) + 3 * max(model.njnt, 1)
               + 2 * max(model.ngeom, 1) + 2 * max(model.nsite, 1))
  float_words = 3 * sum(max(count * width, 1) for count, width in (
      (model.nbody, 3), (model.nbody, 4), (model.njnt, 3),
      (model.njnt, 3), (model.nq, 1), (model.ngeom, 3),
      (model.ngeom, 4), (model.nsite, 3), (model.nsite, 4),
      (model.nbody, 3), (model.nbody, 4)))
  output_words = (6 * (max(2 * model.nbody * 3, 1)
                       + max(2 * model.nbody * 4, 1))
                  + 3 * (max(2 * model.ngeom * 3, 1)
                         + max(2 * model.ngeom * 4, 1)
                         + max(2 * model.ngeom * 9, 1))
                  + 3 * (max(2 * model.nsite * 3, 1)
                         + max(2 * model.nsite * 4, 1))
                  + 2 * max(2 * model.njnt * 3, 1))
  assert runtime["model.fk.static_int"] == int_words * 4
  assert runtime["model.fk.static_float"] == float_words * 4
  assert runtime["fk.pose_output_arena"] == output_words * 4
  assert "model.fk.geom_pos_pair" not in runtime
  assert runtime["flex_contact.vertex_low_tail_dispatch"] == 0
  assert runtime["flex_contact.zero_vertex_residual"] == 0
  assert runtime["flex_contact.zero_geom_residual"] == 0
  assert runtime["passive.fk.auxiliary"] == (
      max(1 + model.nbody + 2 * max(model.ntree, 1) + 2, 1) * 4)
  assert runtime["transmission.position_length"] == (
      2 * max(model.nu, 1) * 4)
  assert runtime["transmission.dims"] == (8 + 2) * 4
  assert runtime["transmission.position_dims"] == (4 + 2) * 4
  assert runtime["passive.projection_dims"] == (4 + 2) * 4
  assert runtime["passive.gravcomp_dims"] == (4 + 2) * 4
  assert runtime["transmission.cached_dims"] == 8 * 4
  assert runtime["transmission.last_qfrc"] == 2 * model.nv * 4
  assert runtime["transmission.length_map"] == max(model.nu * model.nq, 1) * 4
  spatial_nv = max(model.nv, 1)
  assert runtime["spatial_plugin_force.chain"] == 2 * model.nbody * model.nv * 4
  assert runtime["spatial_plugin_force.roots"] == 2 * model.nbody * 4
  assert runtime["spatial_plugin_force.mask"] == 2 * 2 * 4
  assert runtime["spatial_plugin_force.dimensions"] == 2 * 6 * 4
  assert runtime["spatial_plugin_force.output"] == 2 * 2 * spatial_nv * 4
  assert runtime["spatial_plugin_force.query_parameters"] == 2 * 3 * 4
  assert runtime["spatial_plugin_force.magnetic_field"] == 3 * 4
  assert runtime["spatial_plugin_force.site_feedback_parameters"] == 14 * 4
  assert runtime["spatial_plugin_force.magnetic_body_ids"] == max(
      model.nbody - 1, 1) * 4
  assert runtime["sensor.sensor_meta"] == max(model.nsensor, 1) * 10 * 4
  assert runtime["sensor.subtree_runtime"] == (
      4 + 2 * model.nbody * 32 + 2 * max(model.nsensor, 1)) * 4
  from mujoco_metal.model import actuator_tendon_inheritance
  from mujoco_metal.active_contact_links import equality_tree_links
  from mujoco_metal.smooth_metal import compile_tree_mass_layout
  armature, _, _ = actuator_tendon_inheritance(model)
  layout = compile_tree_mass_layout(
      model.body_treeid, model.dof_bodyid,
      tendon_treeid=model.tendon_treeid,
      tendon_treenum=model.tendon_treenum,
      tendon_armature=np.asarray(model.tendon_armature) + armature,
      tendon_j_rowadr=model.ten_J_rowadr,
      tendon_j_rownnz=model.ten_J_rownnz,
      tendon_j_colind=model.ten_J_colind,
      mass_rowadr=model.M_rowadr, mass_rownnz=model.M_rownnz,
      mass_colind=model.M_colind)
  # Simulation defaults to dense SmoothDynamics; sparse allocations are
  # checked in a separately selected exact-storage estimate below.
  assert runtime["smooth.mass_blocks"] == 0
  assert runtime["smooth.mass_zero_blocks"] == 2 * layout["nnz"] * 4
  assert runtime["smooth.mass_armature"] == 2 * model.nv * model.nv * 4
  assert runtime["smooth.pose_status"] == 2 * 4
  assert runtime["model.smooth.component_layout"] == max(
      layout["component_packed"].size, 1) * 4
  assert runtime["component_mass_factor"] == max(2 * layout["nnz"], 1) * 4
  assert runtime["component_mass_factor_status"] == (
      2 * max(layout["ncomponent"], 1) * 4)
  assert runtime["component_mass_factor_dof_mask"] == (
      2 * max(model.nv, 1) * 4)
  assert runtime["component_mass_zero_blocks"] == max(2 * layout["nnz"], 1) * 4
  assert runtime["component_mass_output_pair"] == (
      2 * 2 * (20 + 1) * max(model.nv, 1) * 4)
  assert runtime["component_mass_single_output_pair"] == (
      2 * 2 * max(model.nv, 1) * 4)
  assert runtime["component_smooth_acceleration_pair"] == (
      2 * model.nv * 2 * 4)
  assert runtime["component_mass_diagonal_add"] == (
      2 * max(model.nv, 1) * 4)
  assert runtime["component_mass_damping_mask"] == max(model.nv, 1) * 4
  assert runtime["component_mass_euler_dt"] == 4
  assert runtime["component_mass_allow_indefinite"] == 4
  assert runtime["component_mass_phase"] == 4
  assert runtime["component_mass_layout"] == (
      layout["ncomponent"] + 1 + max(model.nv, 1)
      + max(layout["ncomponent"], 1)) * 4
  assert runtime["sleep.initial_tree_state"] == max(model.ntree, 1) * 4
  assert runtime["sleep.tree_state"] == 2 * max(model.ntree, 1) * 4
  eq_tree_links, _ = equality_tree_links(model)
  assert runtime["sleep.contact_equality_links"] == max(
      (3 + len(eq_tree_links)) * 2 * 2, 1) * 4
  assert runtime["sensor_sleep.treeids"] == max(model.nsensor * 4, 1) * 4
  assert runtime["sensor_sleep.output"] == max(2 * model.nsensor, 1) * 4
  # Four independent trees keep the exact component-factor storage sparse.
  assert layout["nnz"] < model.nv * model.nv
  sparse_estimate = estimate_capacity(
      model, 2, npairs=3, nslots=5, nr=20, mass_storage="block_sparse")
  assert sparse_estimate.mass_storage == "block_sparse"
  sparse_runtime = dict(sparse_estimate.memory_breakdown[
      len(estimate_workspace_expected(model, 2, 3, 5, 20,
                                      mass_storage="block_sparse")):])
  assert sparse_runtime["smooth.mass"] == 0
  assert sparse_runtime["smooth.mass_blocks"] == 2 * layout["nnz"] * 4
  assert sparse_runtime["smooth.mass_armature"] == 2 * layout["nnz"] * 4
  # This fixture uses MuJoCo's default CG solver, so the PGS smooth-force pair
  # is not allocated in either dense or block-sparse mass mode.
  assert sparse_runtime["component_smooth_acceleration_pair"] == 0
  solver_breakdown = dict(sparse_estimate.memory_breakdown)
  sparse_solver = dict(estimate_workspace_expected(
      model, 2, 3, 5, 20, mass_storage="block_sparse"))
  layout_words = 2 * layout["ncomponent"] + 1 + 3 * model.nv
  assert solver_breakdown["workspace_debug"] == (
      sparse_solver["workspace_debug"]
      + 2 * (layout["nnz"] + layout_words) * 4)


def test_component_pgs_capacity_counts_smooth_acceleration_pair():
  model = _model()
  model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
  batch = 2
  estimate = estimate_capacity(
      model, batch, npairs=0, nslots=0, nr=0, mass_storage="block_sparse")
  inventory = dict(estimate.memory_breakdown)
  assert inventory["component_smooth_acceleration_pair"] == (
      2 * batch * int(model.nv) * 4)


def test_dense_pgs_capacity_counts_smooth_force_pair():
  model = _model()
  model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
  batch = 3
  estimate = estimate_capacity(
      model, batch, npairs=0, nslots=0, nr=0, mass_storage="dense")
  inventory = dict(estimate.memory_breakdown)
  assert inventory["component_smooth_acceleration_pair"] == (
      2 * batch * int(model.nv) * 4)


def test_sleep_capacity_includes_static_flex_wake_link_storage():
  from mujoco_metal.active_contact_links import equality_tree_links
  from mujoco_metal.capacity import estimate_capacity
  from mujoco_metal.flex_contact import lower_flex_contacts

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0"><flag sleep="enable"/></option>
    <worldbody>
      <body name="sphere" pos="1 1 1" sleep="init">
        <freejoint/><geom name="ball" type="sphere" size=".07"
          contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="sheet" type="grid" count="2 2 1" pos="0 0 .05"
        spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
        <contact contype="1" conaffinity="0" selfcollide="none" condim="1"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01" elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>''')
  flex_links = int(lower_flex_contacts(model).link_capacity)
  eq_links, _ = equality_tree_links(model)
  estimate = estimate_capacity(
      model, 2, npairs=int(model.npair), nslots=0, nr=0)
  runtime = dict(estimate.memory_breakdown)
  expected_capacity = max(int(model.npair) + len(eq_links) + flex_links, 1)
  assert runtime["sleep.contact_equality_links"] == (
      2 * expected_capacity * 2 * 4)


def test_full_runtime_inventory_names_are_unique_for_sensor_sleep_tendon_model():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option><flag sleep="enable"/></option>
    <worldbody><body><joint name="j" type="slide"/>
      <geom type="sphere" size=".1"/></body></worldbody>
    <tendon><fixed name="t" limited="false" armature=".1">
      <joint joint="j" coef="1"/></fixed></tendon>
    <sensor><jointpos joint="j"/></sensor></mujoco>''')
  parts = _runtime_buffer_sizes(
      model, 3, npairs=int(model.npair), rhs_capacity=max(int(model.nv), 1),
      mass_storage="block_sparse")
  names = [name for name, _ in parts]
  assert len(names) == len(set(names))
  inventory = dict(parts)
  assert inventory["simulation.component_solution_vector"] == (
      3 * model.nv)
  assert inventory["simulation.component_solution_low_vector"] == (
      3 * model.nv)
  assert inventory["simulation.component_solve_rhs"] == (
      3 * max(int(model.nv), 1) * model.nv)
  assert inventory["simulation.fixed_tendon.last_length"] == 3 * model.ntendon
  assert inventory["simulation.fixed_tendon.dims"] == 8 + 3
  assert inventory["simulation.fixed_tendon.velocity_dims"] == 8 + 3
  assert inventory["simulation.position_cache.fixed_tendon_length"] == (
      3 * max(model.ntendon, 1))
  assert inventory["simulation.position_cache.spatial_tendons.jacobian"] == (
      3 * max(model.ntendon, 1) * max(model.nv, 1))
  assert inventory["simulation.position_cache.actuators.moment"] == (
      3 * max(model.nu, 1) * max(model.nv, 1))
  assert inventory["sleep.initial_tree_state"] == max(model.ntree, 1)
  assert inventory["sensor_sleep.treeids"] == max(model.nsensor * 4, 1)
  assert inventory["component_mass_merge_dims"] == 2
  assert inventory["state.reset.clear_rows_dimensions"] == 3
  assert inventory["simulation.component_tendon_jacobian"] == (
      3 * max(model.ntendon, 1) * max(model.nv, 1))
  assert inventory["simulation.component_damping_derivative"] == 3 * model.nv
  assert inventory["simulation.component_world_status"] == 3
  assert inventory["smooth.position_cache.mass_blocks"] == (
      3 * max(model.nv, 1))
  assert inventory["smooth.position_cache.tendon_armature_blocks"] == (
      3 * max(model.nv, 1))
  assert inventory["passive.body_parentid"] == model.nbody
  assert inventory["transmission.moment_map"] == max(model.nu * model.nv, 1)


def test_spatial_tendon_actuator_recovery_rows_are_budgeted():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody>
      <body name="a"><joint type="slide" axis="1 0 0"/>
        <site name="s0"/><geom type="sphere" size=".1"/></body>
      <body name="b" pos="0 1 0"><joint type="slide" axis="0 1 0"/>
        <site name="s1"/><geom type="sphere" size=".1"/></body>
    </worldbody>
    <tendon><spatial name="path"><site site="s0"/><site site="s1"/>
    </spatial></tendon>
    <actuator><general name="path_motor" tendon="path"/></actuator>
  </mujoco>''')
  parts = dict(_runtime_buffer_sizes(
      model, 3, npairs=int(model.npair), rhs_capacity=max(int(model.nv), 1),
      mass_storage="block_sparse"))
  assert parts["actuator.plugin_qfrc_dims"] == 3 + 3
  assert parts["spatial_tendon.actuator_state_rows"] == 3 * max(int(model.nu), 1)
  assert parts["spatial_tendon.actuator_state_dims"] == 5 + 3


def test_legacy_profile_rows_are_counted_as_distinct_real_backings():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint type="slide"/><geom type="sphere" size=".1"/></body>
  </worldbody></mujoco>''')
  batch = 3
  inventory = dict(_runtime_buffer_sizes(
      model, batch, legacy_contact_pairs=2, legacy_joint_rows=4,
      legacy_total_rows=9))
  assert inventory["legacy.contact.canonical_rows"] == (
      max(batch * 2 * 5 * (model.nv + 5), 1))
  assert inventory["legacy.joint.canonical_rows"] == (
      batch * max(4, 1) * (model.nv + 5))
  assert inventory["legacy.joint.lambda"] == batch * max(4, 1)
  assert inventory["simulation.legacy_canonical_rows"] == (
      batch * 9 * (model.nv + 5))
  assert inventory["simulation.legacy_constraint_rhs"] == batch * model.nv
  no_contact_rows = dict(_runtime_buffer_sizes(
      model, batch, legacy_contact_pairs=0))
  assert no_contact_rows["legacy.contact.canonical_rows"] == 1

  # Omitting profile-specific counts must not charge integrated profiles for
  # these separate workspaces.
  generic = dict(_runtime_buffer_sizes(model, batch))
  assert not any(name.startswith("legacy.") for name in generic)
  assert "simulation.legacy_canonical_rows" not in generic
  assert "simulation.legacy_constraint_rhs" not in generic


def test_legacy_profile_row_address_guard_precedes_any_device_allocation():
  from mujoco_metal.capacity import validate_runtime_buffers

  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
  with pytest.raises(CapacityOverflow, match="legacy.joint.canonical_rows.*signed"):
    validate_runtime_buffers(
        model, 1, legacy_joint_rows=(1 << 30))
  with pytest.raises(ValueError, match="legacy_total_rows must be an integer"):
    validate_runtime_buffers(model, 1, legacy_total_rows=3.0)


def test_estimate_capacity_includes_optional_legacy_producer_and_simulation_rows():
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint type="slide"/><geom type="sphere" size=".1"/></body>
  </worldbody></mujoco>''')
  estimate = estimate_capacity(
      model, 2, npairs=1, nslots=1, nr=6,
      legacy_contact_pairs=1, legacy_total_rows=5)
  memory = dict(estimate.memory_breakdown)
  assert memory["legacy.contact.canonical_rows"] == (
      2 * 5 * (model.nv + 5) * 4)
  assert memory["simulation.legacy_canonical_rows"] == (
      2 * 5 * (model.nv + 5) * 4)
  assert "legacy.joint.canonical_rows" not in memory


def test_effective_sparse_implicit_workspace_is_inventory_only_when_requested():
  from mujoco_metal.implicit_effective import effective_gmres_workspace_sizes

  model = _model("slide-chain")
  nv, edge_count, batch = int(model.nv), 13, 2
  estimate = estimate_capacity(
      model, batch, npairs=0, nslots=0, nr=0, mass_storage="block_sparse",
      effective_edge_count=edge_count)
  memory = dict(estimate.memory_breakdown)
  expected = effective_gmres_workspace_sizes(
      batch=batch, nv=nv, edge_count=edge_count)
  for name, elements in expected.items():
    assert memory[f"implicit_effective.{name}"] == elements * 4
  assert memory["velocity_derivative.diagonal_slots"] == max(nv, 1) * 4
  assert memory["velocity_derivative.dims"] == 3 * 4
  assert memory["velocity_derivative.column_dims"] == 4 * 4
  assert memory["velocity_derivative.symmetric_lower_source_slots"] == max(edge_count, 1) * 4
  assert memory["velocity_derivative.symmetric_values"] == batch * max(edge_count, 1) * 4
  assert memory["smooth.bias_low"] == batch * nv * 4
  assert memory["smooth.cvel_low"] == batch * model.nbody * 6 * 4
  assert memory["smooth.cdof_dot_low"] == batch * nv * 6 * 4
  assert memory["smooth.cacc_low"] == batch * model.nbody * 6 * 4
  assert memory["smooth.body_force_low"] == batch * model.nbody * 6 * 4
  assert memory["smooth.bias_coo_column_offsets"] == (nv + 1) * 4
  assert memory["smooth.bias_coo_edge_rows"] == max(edge_count, 1) * 4
  assert memory["smooth.bias_coo_edge_slots"] == max(edge_count, 1) * 4
  assert memory["fixed_tendon.coo_damping_dims"] == 5 * 4
  ordinary = estimate_capacity(
      model, batch, npairs=0, nslots=0, nr=0, mass_storage="block_sparse")
  assert not any(name.startswith("implicit_effective.")
                 for name, _ in ordinary.memory_breakdown)
  # The edge-count gate is host-only and rejects malformed values before any
  # workspace is constructed.
  from mujoco_metal.capacity import validate_runtime_buffers
  with pytest.raises(ValueError, match="effective_edge_count must be an integer"):
    validate_runtime_buffers(model, batch, effective_edge_count=1.5)


def test_sparse_derivative_inventory_includes_owned_actuator_and_fluid_scratch():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option density="1"/>
    <worldbody><body><joint name="slide" type="slide"/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody>
    <actuator><motor joint="slide"/></actuator></mujoco>''')
  batch, edge_count = 3, int(model.nv)
  inventory = dict(_runtime_buffer_sizes(
      model, batch, mass_storage="block_sparse",
      effective_edge_count=edge_count))
  assert inventory["actuator.velocity_derivative_pattern"] == 1
  assert inventory["actuator.velocity_derivative_edge_rows"] == max(edge_count, 1)
  assert inventory["actuator.velocity_derivative_edge_cols"] == max(edge_count, 1)
  assert inventory["actuator.velocity_derivative_dims"] == 8 + batch
  assert inventory["actuator.transmission_kinematics_dims"] == 8 + batch
  assert inventory["actuator.force_dot_dims"] == 5 + batch
  assert inventory["actuator.force_dims"] == 7 + batch
  assert inventory["actuator.qfrc_dims"] == 4 + batch
  assert inventory["actuator.velocity_derivative_scratch"] == (
      batch * max(edge_count, 1))
  for name in ("fluid.derivative_base", "fluid.derivative_velocity",
               "fluid.derivative_column"):
    assert inventory[name] == batch * int(model.nv)


def test_sparse_flex_implicit_inventory_is_linear_and_conditional():
  from mujoco_metal.capacity import _runtime_buffer_sizes
  from mujoco_metal.flex_implicit import requires_flex_implicit_correction
  from mujoco_metal.sparse_flex_implicit import (
      sparse_flex_correction_workspace_sizes)

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option integrator="implicit"/>
    <worldbody><flexcomp name="volume" type="grid" count="2 2 2"
      spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <edge stiffness="0" damping="0"/>
      <elasticity young="1000" poisson=".2" damping=".1"/>
    </flexcomp></worldbody></mujoco>''')
  assert requires_flex_implicit_correction(model)
  batch, edge_count = 2, int(model.nv)
  inventory = dict(_runtime_buffer_sizes(
      model, batch, mass_storage="block_sparse",
      effective_edge_count=edge_count))
  flex_csr_nnz = int(np.asarray(model.flexedge_J_colind).size)
  expected_flex_static = {
      "flex.edge_j_rowadr": int(model.nflexedge),
      "flex.edge_j_rownnz": int(model.nflexedge),
      "flex.edge_j_colind": max(flex_csr_nnz, 1),
      "flex.edge_j_mask_dims": 4,
      "flex.edge_spring_coeff": int(model.nflexedge),
      "flex.edge_operator_coeff": int(model.nflexedge),
      "flex.edge_operator_coo_dims": 4,
  }
  for name, count in expected_flex_static.items():
    assert inventory[name] == count
  expected = sparse_flex_correction_workspace_sizes(
      batch, model.nv, edge_count=edge_count)
  for name, count in expected.items():
    assert inventory[name] == count
  without_sparse_profile = dict(_runtime_buffer_sizes(model, batch))
  assert not any(name.startswith("sparse_flex.")
                 for name in without_sparse_profile)
  for name, count in expected_flex_static.items():
    assert without_sparse_profile[name] == count


@pytest.mark.parametrize("bad", [True, 1.5, -1])
def test_solver_scratch_public_dimensions_reject_bool_fractional_and_negative(bad):
  from mujoco_metal.capacity import primal_scratch_floats, solver_debug_layout

  with pytest.raises(ValueError):
    primal_scratch_floats(bad, 0, int(mujoco.mjtSolver.mjSOL_PGS))
  with pytest.raises(ValueError):
    solver_debug_layout(0, bad, int(mujoco.mjtSolver.mjSOL_PGS))
  with pytest.raises(ValueError, match="solver_type must be an integer"):
    primal_scratch_floats(0, 0, True)


def test_cached_constraint_position_backings_have_exact_capacity_rows():
  from mujoco_metal.capacity import estimate_workspace
  parts, _ = estimate_workspace(
      nv=4, npairs=2, nslots=3, nr=5, nr_joint=1, neq=0, batch=2,
      nbody=2, nsite=1)
  sizes = dict(parts)
  assert sizes["position_context"] == 2 * 5 * 5 * 4
  assert sizes["position_context_zero"] == 2 * 5 * 4
  assert sizes["position_assembly_context"] == 2 * 5 * 5 * 4
  assert sizes["position_surface_velocity"] == 2 * 5 * 4
  assert sizes["position_extra_aref"] == 2 * 5 * 4
  assert sizes["position_refresh_extra_aref"] == 2 * 5 * 4
  assert sizes["position_qvel"] == 2 * 4 * 4
  assert sizes["position_current_qvel"] == 2 * 4 * 4
  # Per-world typed packed-J header is part of the persistent query cache.
  assert sizes["position_cache_J"] == 2 * (12 + 5 * 4) * 4
  assert sizes["position_cache_rows"] == 2 * 7 * 5 * 4
  assert sizes["position_cache_impedance"] == 2 * 5 * 4
  assert sizes["position_cache_contact_data"] == 2 * 3 * 36 * 4
  assert sizes["position_cache_contact_frame"] == 2 * 3 * 12 * 4
  assert sizes["position_cache_contact_jacobian"] == 2 * 3 * 6 * 4 * 4
  assert sizes["position_cache_cvel"] == 2 * 2 * 6 * 4
  assert sizes["position_cache_cdof"] == 2 * 4 * 6 * 4
  assert sizes["position_cache_cdof_dot"] == 2 * 4 * 6 * 4
  assert sizes["position_current_cvel"] == 2 * 2 * 6 * 4
  assert sizes["position_current_cdof"] == 2 * 4 * 6 * 4
  assert sizes["position_current_cdof_dot"] == 2 * 4 * 6 * 4
  assert sizes["position_current_body_pos"] == 2 * 2 * 3 * 4
  assert sizes["position_current_body_quat"] == 2 * 2 * 4 * 4
  assert sizes["position_current_root_com"] == 2 * 2 * 3 * 4
  assert sizes["position_current_site_pos"] == 2 * 1 * 3 * 4
  assert sizes["position_current_site_quat"] == 2 * 1 * 4 * 4
  assert sizes["equality_pose_com"] == 2 * 2 * 2 * 3 * 4
  assert sizes["equality_motion"] == 2 * 4 * 12 * 4
  assert sizes["position_cache_body_pos"] == 2 * 2 * 3 * 4
  assert sizes["position_cache_body_quat"] == 2 * 2 * 4 * 4
  assert sizes["position_cache_root_com"] == 2 * 2 * 3 * 4
  assert sizes["position_cache_site_pos"] == 2 * 1 * 3 * 4
  assert sizes["position_cache_site_quat"] == 2 * 1 * 4 * 4
  assert sizes["position_cache_slot_packed"] == 2 * 3 * 4
  assert sizes["position_cache_slot_count"] == 2 * 4
  assert sizes["position_cache_pair_packed"] == 2 * 2 * 4
  assert sizes["position_cache_pair_overflow"] == 2 * 4
  empty, _ = estimate_workspace(
      nv=0, npairs=0, nslots=0, nr=0, nr_joint=0, neq=0, batch=1)
  assert dict(empty)["position_context"] == 4


def test_signed_euler_ldl_awake_workspace_inventory_matches_prepared_shapes():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint type="slide" damping=".5"/>
      <geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
  sizes = dict(_runtime_buffer_sizes(model, 3))
  assert sizes["simulation.euler_awake.dof_ids"] == 3 * max(model.nv, 1)
  assert sizes["simulation.euler_awake.counts"] == 3 * 3
  assert sizes["simulation.euler_solver.factor"] == 3 * model.nv * model.nv
  assert sizes["simulation.euler_solver.solution"] == 3 * model.nv
  assert sizes["simulation.euler_solver.status"] == 3
  assert sizes["simulation.euler_solver.empty_input"] == 1
  assert sizes["simulation.euler_solver.dims"] == 3
  assert sizes["simulation.euler_solver.awake_work"] == 3 * model.nv
  assert sizes["simulation.euler_solver.awake_dims"] == 3
  assert sizes["simulation.euler_solver.awake_flags"] == 2
  assert sizes["simulation.rhs"] == 3 * model.nv
  assert sizes["simulation.rhs_low"] == 3 * model.nv
  assert sizes["simulation.dense_pair_solution_low"] == 3 * model.nv
  assert sizes["simulation.dense_pair_zero_low"] == 3 * model.nv
  assert sizes["simulation.dense_pair_work"] == 18 * max(model.nv, 1)
  assert sizes["simulation.solver_fwdinv"] == 3 * 2
  assert sizes["simulation.effective_mass"] == 3 * model.nv * model.nv
  assert sizes["simulation.euler_damping_eligibility"] == max(model.nv, 1)


def test_rk4_initial_velocity_snapshot_is_counted_only_for_rk4():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint type="slide"/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
  rk4_sizes = dict(_runtime_buffer_sizes(model, 3))
  assert rk4_sizes["rk4.initial_qvel"] == 3 * model.nv
  assert rk4_sizes["rk4.zero"] == 3 * model.nv
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
  euler_sizes = dict(_runtime_buffer_sizes(model, 3))
  assert "rk4.initial_qvel" not in euler_sizes
  assert "rk4.zero" not in euler_sizes


def test_runtime_inventory_handles_empty_passive_joint_arrays():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><geom type="plane" size="2 2 .1"/></worldbody>
  </mujoco>''')
  inventory = dict(_runtime_buffer_sizes(model, 2))
  assert model.njnt == 0
  assert inventory["passive.springpoly"] == 1
  assert inventory["passive.damping"] == 1
  assert inventory["passive.fk.qpos"] == 1


def test_passive_and_transmission_exact_backing_guards_are_preallocation():
  from types import SimpleNamespace
  from mujoco_metal.passive import passive_workspace_sizes
  from mujoco_metal.transmissions import transmission_workspace_sizes

  # Synthetic immutable metadata lets the test cross a single addressed
  # backing limit without trying to construct or allocate a huge model.
  passive_meta = SimpleNamespace(
      nq=1, nv=50_000, njnt=1, nbody=2, ngeom=0, nsite=0, nmocap=0,
      nu=0, ntendon=0, ten_J_colind=np.zeros(0, dtype=np.int32),
      M_colind=np.zeros(50_000, dtype=np.int32),
      body_treeid=np.asarray([0, 0], dtype=np.int32),
      jnt_stiffnesspoly=np.zeros((1, 2), dtype=np.float32))
  with pytest.raises(ValueError, match="passive.force.*int32 address range"):
    passive_workspace_sizes(passive_meta, 50_000)

  transmission_meta = SimpleNamespace(nq=50_000, nv=1, nu=50_000)
  with pytest.raises(ValueError, match="transmission.length_map.*int32 address range"):
    transmission_workspace_sizes(transmission_meta, 1)


def test_default_limits_cover_source_contact_pair_capacity():
  assert (BASE_NVIDIA_NV, BASE_MAX_PAIRS, BASE_MAX_SLOTS, BASE_MAX_ROWS) == (32, 32, 50, 640)
  lim = CapacityLimits()
  assert (lim.max_nv, lim.max_pairs, lim.max_slots, lim.max_rows) == (32, 32, 50, 640)


def test_overflow_messages_preserve_historical_text():
  import dataclasses
  model = _model()
  with pytest.raises(ValueError, match="bounds nv to 32"):
    check_capacity(dataclasses.replace(estimate_capacity(model, 1, 0, 0, 0), nv=33))
  with pytest.raises(ValueError, match=r"candidate contact pairs \(33\) exceeds capacity 32"):
    check_capacity(estimate_capacity(model, 1, 33, 0, 0))
  with pytest.raises(ValueError, match=r"total candidate contact slots \(49\) exceeds capacity 48"):
    check_capacity(estimate_capacity(model, 1, 0, 49, 0),
                   CapacityLimits(max_slots=48))
  with pytest.raises(ValueError, match=r"total candidate constraint rows \(97\) exceeds capacity 96"):
    check_capacity(estimate_capacity(model, 1, 0, 0, 97),
                   CapacityLimits(max_rows=96))


def test_user_limits_and_memory_budget_are_honored():
  model = _model()
  est = estimate_capacity(model, 1, 2, 2, 10)
  tight = CapacityLimits(max_pairs=1)
  with pytest.raises(CapacityOverflow, match="pairs"):
    check_capacity(est, tight)
  tiny_mem = CapacityLimits(memory_budget_bytes=1)
  with pytest.raises(CapacityOverflow, match="memory"):
    check_capacity(est, tiny_mem)
  assert check_capacity(est) is est


def test_capacity_limits_validate_gmres_restart_and_iteration_budgets():
  import dataclasses
  estimate = dataclasses.replace(
      estimate_capacity(_model(), 1, 0, 0, 0), nv=3)
  assert check_capacity(
      estimate, CapacityLimits(gmres_krylov_dimension=2,
                               gmres_max_iterations=9)) is estimate
  for kwargs in (
      {"gmres_krylov_dimension": True},
      {"gmres_krylov_dimension": 1.5},
      {"gmres_krylov_dimension": 4},
      {"gmres_max_iterations": False},
      {"gmres_max_iterations": 0},
  ):
    with pytest.raises(CapacityOverflow, match="gmres_"):
      check_capacity(estimate, CapacityLimits(**kwargs))


def test_explicit_user_limits_set_dynamic_solver_dimensions():
  import dataclasses
  model = _model()
  estimate = estimate_capacity(model, 1, 0, 0, 0)
  requested_nv, requested_rows = 73, 261
  raised = dataclasses.replace(estimate, nv=requested_nv, nr=requested_rows)
  assert check_capacity(
      raised, CapacityLimits(max_nv=requested_nv,
                             max_rows=requested_rows)) is raised
  with pytest.raises(ValueError, match=f"bounds nv to {requested_nv - 1}"):
    check_capacity(
        raised, CapacityLimits(max_nv=requested_nv - 1,
                               max_rows=requested_rows))
  with pytest.raises(ValueError, match=f"capacity {requested_rows - 1}"):
    check_capacity(
        raised, CapacityLimits(max_nv=requested_nv,
                               max_rows=requested_rows - 1))


def test_overflow_is_valueerror_and_batch_checked():
  model = _model()
  est = estimate_capacity(model, 17, 0, 0, 0)
  with pytest.raises(ValueError, match="batch"):
    check_capacity(est)
  with pytest.raises(TypeError):
    estimate_capacity("not-a-model", 1, 0, 0, 0)
  with pytest.raises(ValueError, match="positive"):
    estimate_capacity(model, 0, 0, 0, 0)


def test_signed_int32_buffer_addressing_precedes_user_memory_budget():
  # Arithmetic only: these dimensions must fail before any device allocation
  # even when a caller configures a very large byte budget.
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
  with pytest.raises(CapacityOverflow, match="workspace_debug.*signed 32-bit"):
    estimate_capacity(model, 32768, npairs=0, nslots=0, nr=256)

  with pytest.raises(CapacityOverflow, match="contact_jacobian.*signed 32-bit"):
    estimate_workspace(
        nv=64, npairs=0, nslots=100, nr=1, nr_joint=0, neq=0,
        batch=100000, solver_type=int(mujoco.mjtSolver.mjSOL_PGS))
  _check_i32_elements("boundary", SIGNED_I32_ELEMENT_LIMIT)
  with pytest.raises(CapacityOverflow, match="boundary.*signed 32-bit"):
    _check_i32_elements("boundary", SIGNED_I32_ELEMENT_LIMIT + 1)


@pytest.mark.parametrize("field,value", [
    ("batch", True), ("batch", 1.5), ("nv", -1), ("npairs", 1.5),
    ("nslots", -1), ("nr", True), ("nr_joint", -1), ("neq", 2.0),
])
def test_workspace_dimensions_reject_bool_fractional_and_negative(field, value):
  args = dict(nv=1, npairs=0, nslots=0, nr=0, nr_joint=0, neq=0, batch=1)
  args[field] = value
  with pytest.raises(ValueError):
    estimate_workspace(**args)


def test_public_capacity_batch_rejects_fractional_and_boolean_values():
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
  for batch in (True, 1.5, 0, -1):
    with pytest.raises(ValueError):
      estimate_capacity(model, batch, 0, 0, 0)


def estimate_workspace_expected(model, batch, npairs, nslots, nr,
                                *, mass_storage="dense"):
  """Independent exact byte count for the allocated solver/map buffers."""
  b, v = batch, max(int(model.nv), 1)
  nc = nslots
  sizes = (
      max(b * nc * 36, 1), max(b * nc * 12, 1),
      max(b * nc * 6 * v, 1), max(b * max(npairs, 1), 1),
      max(b * (12 + nr * v), 1), max(b * nr, 1), max(b * (nr * nr + 7 * nr
                                    + primal_scratch_floats(
                                        v, nr, model.opt.solver, model.ntree,
                                        model.nbody, mass_storage)
                                    + 8 * v + 3 * max(nr, 1)), 1),
      22 + 6 + v + 10 * nr + 20,
      max(b * v, 1), max(2 * b * int(model.nv), 1), b, max(b * 10, 1),
      max(b * nc * 11, 1), max(b, 1), max(b, 1), max(b, 1),
      b * npairs, b * npairs, b * nc, b * nc, b * nr, b * nr, b * 4,
      b * max(npairs, 1), b * max((npairs + 255) // 256, 1) * 2,
      b * max(nc, 1), b * max((nc + 255) // 256, 1) * 2,
      b * max(nr, 1), b * max((nr + 255) // 256, 1) * 2,
  )
  names = (
      "contact_row_data", "contact_frame", "contact_jacobian", "pair_mask",
      "workspace_J", "position_context_zero", "workspace_debug",
      "solver_dims",
      "out_force", "out_acc", "out_status",
      "out_diagnostics", "out_contact_force", "out_joint_force", "eq_active",
      "eq_active_default", "packed_to_pair", "pair_to_packed", "packed_to_slot",
      "slot_to_packed", "packed_to_logical_row", "logical_to_packed_row",
      "compaction_counts_overflow", "pair_scan_prefix", "pair_scan_blocks",
      "slot_scan_prefix", "slot_scan_blocks", "row_scan_prefix", "row_scan_blocks",
  )
  return tuple(zip(names, (n * 4 for n in sizes)))


def _jacobian_model(nv, setting="auto"):
  bodies = "".join(
      '<body><joint type="slide" axis="1 0 0"/><geom type="sphere" size=".01"/></body>'
      for _ in range(nv))
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option jacobian="{setting}"/><worldbody>{bodies}</worldbody></mujoco>')


@pytest.mark.parametrize("nv, expected", [(60, "dense"), (61, "sparse")])
def test_auto_jacobian_uses_pinned_sixty_dof_dispatch(nv, expected):
  model = _jacobian_model(nv)
  assert selected_jacobian_kind(model) == expected
  estimate = estimate_capacity(model, 1, 0, 0, 0)
  assert estimate.jacobian_kind == expected
  assert estimate.jacobian_auto_threshold == AUTO_JACOBIAN_DENSE_NV == 60


@pytest.mark.parametrize("setting, expected", [("dense", "dense"), ("sparse", "sparse")])
def test_explicit_jacobian_setting_overrides_auto_dispatch(setting, expected):
  model = _jacobian_model(61, setting)
  assert selected_jacobian_kind(model) == expected
  assert estimate_capacity(model, 1, 0, 0, 0).jacobian_kind == expected


def test_energy_constants_are_counted_as_their_actual_separate_backings():
  from mujoco_metal.capacity import _runtime_buffer_sizes
  xml = '''<mujoco><option><flag energy="enable"/></option>
    <worldbody><body><joint type="slide"/>
      <geom type="sphere" size=".1" mass="1"/>
    </body></worldbody></mujoco>'''
  enabled = mujoco.MjModel.from_xml_string(xml)
  sizes = dict(_runtime_buffer_sizes(enabled, 2))
  assert sizes["simulation.energy_stage"] == 2 * 2
  assert sizes["simulation.energy_body_mass_const"] == max(enabled.nbody - 1, 1)
  assert sizes["simulation.energy_gravity_const"] == 3
  assert sizes["simulation.energy_qpos_spring_const"] == max(enabled.nq, 1)
  assert sizes["simulation.energy_position_dims"] == 12 + 2
  assert sizes["simulation.energy_joint_int"] == 3 * max(enabled.njnt, 1)
  assert sizes["simulation.energy_tendon_int"] == 4 * max(enabled.ntendon, 1)
  disabled = mujoco.MjModel.from_xml_string(xml.replace(
      '<flag energy="enable"/>', '<flag energy="disable"/>'))
  disabled_sizes = dict(_runtime_buffer_sizes(disabled, 2))
  assert disabled_sizes["simulation.energy_stage"] == 0
  assert disabled_sizes["simulation.energy_body_mass_const"] == 0
  assert disabled_sizes["simulation.energy_gravity_const"] == 0
  assert disabled_sizes["simulation.energy_qpos_spring_const"] == 0
  assert disabled_sizes["simulation.energy_position_dims"] == 0
  assert disabled_sizes["simulation.energy_joint_int"] == 0
