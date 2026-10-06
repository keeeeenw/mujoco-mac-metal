# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracle for pinned MuJoCo 3.10 PGS block visitation order."""

import mujoco
import numpy as np
import os
import pytest

from mujoco_metal.capacity import primal_scratch_floats
from mujoco_metal.coupled_constraints import _position_reference_aref_cpu


_PCG_MULTIPLIER = 6364136223846793005
_UINT32_MASK = (1 << 32) - 1
_UINT64_MASK = (1 << 64) - 1


def _pinned_sweeps(count, sweeps):
  """Mirror engine_solver.c's seeded PCG32 + in-place Fisher-Yates shuffle."""
  state, increment = 0, 1

  def next_u32():
    nonlocal state
    old = state
    state = (old * _PCG_MULTIPLIER + (increment | 1)) & _UINT64_MASK
    xorshifted = (((old >> 18) ^ old) >> 27) & _UINT32_MASK
    rotation = old >> 59
    return ((xorshifted >> rotation)
            | (xorshifted << ((-rotation) & 31))) & _UINT32_MASK

  next_u32()  # pinned solver advances once after initializing its seed
  order = list(range(count))
  result = []
  for _ in range(sweeps):
    for index in range(count - 1, 0, -1):
      other = next_u32() % (index + 1)
      order[index], order[other] = order[other], order[index]
    result.append(tuple(order))
  return result


def _active_block_order(active, elliptic_blocks):
  """Canonical active constraint rows, grouping only elliptic contacts."""
  active = np.asarray(active, dtype=bool)
  grouped = set()
  starts = {}
  for start, dim in elliptic_blocks:
    assert dim > 1 and start >= 0 and start + dim <= len(active)
    assert not any(row in grouped for row in range(start, start + dim))
    grouped.update(range(start, start + dim))
    starts[start] = dim
  blocks = []
  for row, enabled in enumerate(active):
    if not enabled:
      continue
    if row not in grouped or row in starts:
      blocks.append(row)
  return blocks


def _two_island_contact_model(iterations):
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" solver="PGS"
        cone="pyramidal" iterations="{iterations}" tolerance="1e-12"/>
    <worldbody>
      <geom type="plane" size="2 2 .1" condim="3" friction=".8 .6 .07"/>
      <body name="left" pos="-.5 0 .095"><freejoint/>
        <geom type="sphere" size=".1" mass="1" condim="3" friction=".8 .6 .07"/>
      </body>
      <body name="right" pos=".5 0 .095"><freejoint/>
        <geom type="sphere" size=".1" mass="1" condim="3" friction=".8 .6 .07"/>
      </body>
    </worldbody>
  </mujoco>""")


def test_converged_qacc_seed_is_accepted_by_zero_budget_cpu_pgs():
  """The warm fixture must change the pinned zero-budget returned iterate."""
  model = _two_island_contact_model(0)
  qpos = np.broadcast_to(np.asarray(model.qpos0, dtype=np.float32),
                         (2, model.nq)).copy()
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  model.opt.tolerance = 1e-6
  model.opt.iterations = 200
  warm = np.zeros((2, model.nv), dtype=np.float32)
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    assert data.nisland == 2
    assert data.solver_niter[0] < 200
    warm[world] = data.qacc
  model.opt.iterations = 0
  model.opt.tolerance = 1e-12
  seeded, cold = [], []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.qacc_warmstart[:] = warm[world]
    mujoco.mj_forward(model, data)
    seeded.append(np.asarray(data.qacc).copy())
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    cold.append(np.asarray(data.qacc).copy())
  assert np.linalg.norm(np.asarray(seeded) - np.asarray(cold)) > 1.0


def test_pinned_pgs_pcg32_first_sweeps_across_block_counts():
  # These are source-derived outputs from MuJoCo 3.10 engine_solver.c's
  # state=0/inc=1 initialization and Fisher-Yates loop. The permutation list
  # is mutated across sweeps; the PCG stream is not reseeded each iteration.
  expected = {
      0: [(), (), ()],
      1: [(0,), (0,), (0,)],
      2: [(1, 0), (0, 1), (1, 0)],
      3: [(1, 2, 0), (1, 0, 2), (0, 2, 1)],
      4: [(2, 3, 1, 0), (3, 1, 2, 0), (0, 3, 1, 2)],
      6: [(4, 5, 1, 2, 3, 0), (0, 3, 2, 5, 1, 4),
          (3, 2, 0, 5, 1, 4)],
  }
  for count, schedules in expected.items():
    assert _pinned_sweeps(count, 3) == schedules


def test_pgs_order_filters_inactive_rows_and_groups_elliptic_blocks():
  # Equality/contact activity can remove logical rows from a world. Elliptic
  # friction contributes one PGS block of its full dimensionality; pyramidal
  # rows and all other active scalar constraints each contribute one block.
  active = [True, False, True, True, True, True, True]
  blocks = _active_block_order(active, elliptic_blocks=[(2, 3)])
  assert blocks == [0, 2, 5, 6]
  assert _pinned_sweeps(len(blocks), 1)[0] == (2, 3, 1, 0)


def test_pgs_permutation_workspace_accounts_for_qacc_and_row_order():
  pgs = int(mujoco.mjtSolver.mjSOL_PGS)
  # Includes qacc warmstart, one island row map, int32 DOF/row labels,
  # union-find parent/used/root, midpoint masks, and awake-tree scratch
  # (guarded dimensions when ntree/nbody are omitted).
  # Equality assembly's four 3xnv point-J blocks are reserved in the same
  # world-local tail after the island maps and before retained qacc.
  # 14 PGS row vectors + dense factor/solve (v²+2v), 12 typed metadata
  # vectors, the retained [nr,nv] mass solve, island storage, and qacc tail.
  assert primal_scratch_floats(6, 17, pgs) == 699
  # Empty qacc/constraint dimensions still have independent guarded slots.
  assert primal_scratch_floats(0, 0, pgs) == 51


def test_awake_tree_mask_uses_reserved_scratch_before_qacc_tail():
  pgs = int(mujoco.mjtSolver.mjSOL_PGS)
  nv, nr, ntree, nbody = 6, 17, 3, 5
  total = primal_scratch_floats(nv, nr, pgs, ntree, nbody)
  core = (primal_scratch_floats(nv, nr, pgs, 0, 0)
          - 2 * nv - nr - 6 - 12 * nv)
  awake_start = core + nv + nr + 4 * ntree + nbody
  warmstart_start = total - nv
  assert awake_start == 632
  assert awake_start + ntree + 12 * nv == warmstart_start


def test_solver_dispatch_stage_restores_dimensions_after_delegate_failure():
  from mujoco_metal.coupled_constraints import _solver_dimension_stage

  dims = [0] * 21
  dims[20] = 7
  with pytest.raises(RuntimeError, match="delegate failed"):
    with _solver_dimension_stage(dims, -1, restore_value=7):
      assert dims[20] == -1
      raise RuntimeError("delegate failed")
  assert dims[20] == 7


def test_position_context_refresh_preserves_bias_surface_and_extra_terms():
  J = np.asarray([[[1.0, 2.0], [-1.0, 0.5]]])
  qvel = np.asarray([[0.25, -0.5]])
  context = np.asarray([[[2.0, 3.0, .8, .4, .1],
                        [0.0, 0.0, .6, 1.0, .0]]])
  surface = np.asarray([[.2, -.1]])
  extra = np.asarray([[.7, -.3]])
  actual = _position_reference_aref_cpu(J, qvel, context, surface, extra)
  Jv = np.einsum("brv,bv->br", J, qvel) + surface
  expected = (-context[..., 1] * Jv
              - context[..., 0] * context[..., 2]
              * (context[..., 3] - context[..., 4]) + extra)
  np.testing.assert_allclose(actual, expected)
  np.testing.assert_allclose(actual[0, 1], extra[0, 1])


def test_sparse_primal_solver_requires_configured_operator_before_mutation():
  from types import SimpleNamespace
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  sentinel = object()
  solver = object.__new__(MetalCoupledConstraints)
  solver._workspace = sentinel
  solver._torch = sentinel
  solver._component_operator_layout_device = None
  solver._candidate_generation_token = token = object()
  solver._position_current_valid = True
  solver.descriptor = SimpleNamespace(
      nv=1, solver_type=int(mujoco.mjtSolver.mjSOL_CG),
      ncontacts_max=0, nr=0)
  solver.batch_size = 1
  with pytest.raises(ValueError, match="configured component layout"):
    solver.run_device(
        poses=None, mass=None, qfrc_smooth=None, qpos=None, qvel=None,
        component_solver=object())
  assert solver._workspace is sentinel
  assert solver._candidate_generation_token is token
  assert solver._position_current_valid is True


def test_missing_mass_rejected_before_component_workspace_mutation():
  from types import SimpleNamespace
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  sentinel = object()
  solver = object.__new__(MetalCoupledConstraints)
  solver._workspace = sentinel
  solver._torch = sentinel
  solver._candidate_generation_token = token = object()
  solver._position_current_valid = True
  solver.descriptor = SimpleNamespace(
      nv=1, solver_type=int(mujoco.mjtSolver.mjSOL_PGS),
      ncontacts_max=0, nr=0)
  solver.batch_size = 1
  with pytest.raises(ValueError, match="dense solver path requires a mass matrix"):
    solver.run_device(
        poses=None, mass=None, qfrc_smooth=None, qpos=None, qvel=None)
  assert solver._workspace is sentinel
  assert solver._candidate_generation_token is token
  assert solver._position_current_valid is True


def test_component_assembly_stage_returns_before_dense_mass_factorization():
  from pathlib import Path

  source = (Path(__file__).parents[1]
            / "mujoco_metal" / "shaders" / "coupled_constraints.metal").read_text()
  for kernel_name in ("solve_coupled_constraints(",
                      "solve_coupled_constraints_block("):
    start = source.index("kernel void " + kernel_name)
    body_open = source.index("{", start)
    depth, body_end = 0, None
    for cursor in range(body_open, len(source)):
      if source[cursor] == "{":
        depth += 1
      elif source[cursor] == "}":
        depth -= 1
        if depth == 0:
          body_end = cursor
          break
    assert body_end is not None
    body = source[body_open:body_end]
    assembly_return = body.index("if (dims[20] == -1)")
    first_mass_factor = body.index("solver_factor_component_mass(")
    assert assembly_return < first_mass_factor
    # Assembly-only returns before either dense factorization or the
    # component solve/factor stage. This protects the inverse/stage-1 path.
    assert "mass[mb + i * nv + j]" not in body[:assembly_return]
    assert "if (dims[20] == -2)" in body


def test_pgs_permutation_payload_preserves_full_int32_row_ids():
  # The permutation occupies float scratch, so it stores integer IDs as raw
  # bits. Numeric float conversion would corrupt IDs above 2^24.
  rows = np.asarray([0, (1 << 24) + 7, (1 << 31) - 1], dtype=np.uint32)
  encoded = rows.view(np.float32)
  np.testing.assert_array_equal(encoded.view(np.uint32), rows)
  shader = (__import__("pathlib").Path(__file__).parents[1]
            / "mujoco_metal" / "shaders" / "coupled_constraints.metal")
  text = shader.read_text()
  assert "pgs_order_encode" in text and "as_type<uint>(encoded)" in text


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_pgs_first_sweeps_match_pinned_contact_equality_and_friction(iterations):
  """Active-row filtering and one-island contact blocks follow pinned order."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_setState

  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" solver="PGS"
        cone="pyramidal" iterations="{iterations}" tolerance="1e-12"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" condim="3"
          friction=".8 .6 .07"/>
      <body pos="0 0 .095">
        <freejoint name="root"/>
        <geom name="ball" type="sphere" size=".1" mass="1"
            condim="3" friction=".8 .6 .07"/>
        <body pos="1 0 0">
          <joint name="a" type="slide" axis="1 0 0" frictionloss=".2"/>
          <geom type="sphere" size=".05" mass="1"
              contype="0" conaffinity="0"/>
        </body>
        <body pos="2 0 0">
          <joint name="b" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".05" mass="1"
              contype="0" conaffinity="0"/>
        </body>
      </body>
    </worldbody>
    <equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
  </mujoco>""")
  # All six rows are in one compiled kinematic tree/island. Turning off the
  # equality per world makes its row disappear from the pinned efc list before
  # MuJoCo builds and shuffles that world's PGS blocks.
  qpos = np.broadcast_to(np.asarray(model.qpos0, dtype=np.float32),
                         (3, model.nq)).copy()
  qvel = np.asarray([
      [.3, -.2, -.1, .1, -.2, .5, .4, 0.],
      [-.1, .25, -.3, .2, .4, -.2, 0., -.35],
      [.2, .1, -.15, -.4, .3, .25, -.2, .5],
  ], dtype=np.float32)
  active = np.asarray([[1], [0], [1]], dtype=np.int32)
  reference = []
  for world in range(3):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.eq_active[:] = active[world]
    mujoco.mj_forward(model, data)
    assert data.nisland == 1, data.nisland
    # Equality, dry-friction, and the four pyramidal contact rows are all live
    # in worlds 0/2; only the equality row is filtered out in world 1.
    assert data.nefc == (6 if active[world, 0] else 5), data.nefc
    reference.append(np.asarray(data.qacc).copy())

  sim = MetalSimulation(model, batch_size=3, profile="integrated_euler_v1")
  sim.reset(qpos=qpos, qvel=qvel)
  mj_setState(sim, {"eq_active": active})
  result = sim.assembled_system(recompute=True)
  actual = result["qacc"].detach().cpu().numpy().astype(np.float64)
  np.testing.assert_allclose(actual, np.asarray(reference), rtol=3e-4, atol=3e-3)
  assert torch.isfinite(result["lambda"]).all()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("iterations", [0, 1, 2])
@pytest.mark.parametrize("warm", [False, True])
def test_pgs_per_island_low_budget_and_qacc_warmstart_parity(iterations, warm):
  """Each disconnected contact island starts its own pinned PGS shuffle."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_setState

  model = _two_island_contact_model(iterations)
  qpos = np.broadcast_to(np.asarray(model.qpos0, dtype=np.float32),
                         (2, model.nq)).copy()
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  configured_iterations = int(model.opt.iterations)
  warm_acc = np.zeros((2, model.nv), dtype=np.float32)
  if warm:
    # Derive a genuinely useful acceleration warmstart from a converged CPU
    # solve. A hand-picked 9.81 seed is rejected by MuJoCo's cost check for
    # this contact frame and therefore cannot qualify warmstart consumption.
    configured_tolerance = float(model.opt.tolerance)
    model.opt.tolerance = 1e-6
    model.opt.iterations = max(configured_iterations, 200)
    for world in range(2):
      converged = mujoco.MjData(model)
      converged.qpos[:] = qpos[world]
      converged.qvel[:] = qvel[world]
      mujoco.mj_forward(model, converged)
      warm_acc[world] = converged.qacc
      assert np.linalg.norm(warm_acc[world]) > 1.0
    model.opt.iterations = configured_iterations
    model.opt.tolerance = configured_tolerance

  expected = []
  cold_expected = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    if warm:
      data.qacc_warmstart[:] = warm_acc[world]
    mujoco.mj_forward(model, data)
    assert data.nisland == 2, data.nisland
    assert data.nefc >= 8, data.nefc
    expected.append(np.asarray(data.qacc).copy())
    if warm:
      cold = mujoco.MjData(model)
      cold.qpos[:] = qpos[world]
      cold.qvel[:] = qvel[world]
      mujoco.mj_forward(model, cold)
      cold_expected.append(np.asarray(cold.qacc).copy())
  if warm and iterations == 0:
    assert np.linalg.norm(np.asarray(expected) - np.asarray(cold_expected)) > 1.0

  native = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  native.reset(qpos=qpos, qvel=qvel)
  if warm:
    mj_setState(native, {"qacc_warmstart": warm_acc})
  result = native.assembled_system(recompute=True)
  actual = result["qacc"].detach().cpu().numpy().astype(np.float64)
  np.testing.assert_allclose(actual, np.asarray(expected), rtol=4e-4, atol=4e-3)
  assert torch.isfinite(result["lambda"]).all()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("iterations", [0, 1, 2])
@pytest.mark.parametrize("warm_kind", ["cold", "accepted", "rejected"])
def test_pgs_elliptic_contact_low_budget_matches_pinned(
    dim, iterations, warm_kind):
  """Native PGS retains pinned elliptic contact blocks at finite budgets."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_setState

  def make_model(budget):
    return mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option timestep=".002" gravity="0 0 -9.81" solver="PGS"
          cone="elliptic" impratio="3.5" iterations="{budget}"
          tolerance="1e-12"/>
      <worldbody>
        <geom type="plane" size="2 2 .1" friction=".8 .03 .01"
            condim="{dim}"/>
        <body pos="0 0 .04"><freejoint name="root"/>
          <geom type="sphere" size=".05" mass="1"
              friction=".8 .03 .01" condim="{dim}"/>
        </body>
      </worldbody>
    </mujoco>""")

  qpos = np.asarray([0, 0, .04, 1, 0, 0, 0], dtype=np.float64)
  qvel = np.asarray([.7, -.4, .1, .3, -.2, .5], dtype=np.float64)
  model = make_model(iterations)
  if warm_kind == "accepted":
    seed_model = make_model(200)
    seed_data = mujoco.MjData(seed_model)
    seed_data.qpos[:] = qpos
    seed_data.qvel[:] = qvel
    mujoco.mj_forward(seed_model, seed_data)
    assert seed_data.ncon == 1
    assert seed_data.solver_niter[0] < 200
    warm_seed = np.asarray(seed_data.qacc).copy()
  elif warm_kind == "rejected":
    warm_seed = np.full(model.nv, 1000.0, dtype=np.float64)
  else:
    warm_seed = np.zeros(model.nv, dtype=np.float64)

  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos
  cpu.qvel[:] = qvel
  cpu.qacc_warmstart[:] = warm_seed
  mujoco.mj_forward(model, cpu)
  assert cpu.ncon == 1
  assert cpu.contact[0].dim == dim
  assert np.isfinite(cpu.contact[0].mu)
  if iterations == 0 and warm_kind == "accepted":
    cold = mujoco.MjData(model)
    cold.qpos[:] = qpos
    cold.qvel[:] = qvel
    mujoco.mj_forward(model, cold)
    assert np.linalg.norm(cpu.qacc - cold.qacc) > 1e-4

  native = MetalSimulation(
      model, qpos=qpos[None, :].astype(np.float32),
      qvel=qvel[None, :].astype(np.float32),
      profile="integrated_euler_v1")
  mj_setState(native, {"qacc_warmstart": warm_seed[None, :].astype(np.float32)})
  system = native.assembled_system(recompute=True)
  actual_acc = system["qacc"].detach().cpu().numpy()[0]
  actual_force = system["qfrc_constraint"].detach().cpu().numpy()[0]
  np.testing.assert_allclose(actual_acc, cpu.qacc, atol=4e-3, rtol=4e-4)
  np.testing.assert_allclose(actual_force, cpu.qfrc_constraint,
                             atol=4e-3, rtol=4e-4)
  assert torch.isfinite(system["lambda"]).all()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("iterations", [0, 1, 2])
@pytest.mark.parametrize("dim", [3, 4, 6])
def test_pgs_elliptic_contact_low_budget_warmstart_parity(iterations, dim):
  """Finite-budget elliptic PGS matches three pinned warmstart states."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_setState

  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" solver="PGS"
        cone="elliptic" impratio="3.5" iterations="{iterations}"
        tolerance="1e-12"/>
    <worldbody>
      <geom type="plane" size="2 2 .1" condim="{dim}"
          friction=".8 .03 .01"/>
      <body pos="0 0 .04"><freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="{dim}"
            friction=".8 .03 .01"/>
      </body>
    </worldbody>
  </mujoco>""")
  qpos = np.broadcast_to(np.asarray(model.qpos0, dtype=np.float32),
                         (3, model.nq)).copy()
  qvel = np.asarray([
      [.7, -.4, .1, .3, -.2, .5],
      [-.3, .2, -.1, -.2, .4, -.5],
      [.1, .35, -.25, .4, -.3, .2],
  ], dtype=np.float32)
  configured_iterations = int(model.opt.iterations)
  model.opt.iterations = max(configured_iterations, 100)
  accepted = mujoco.MjData(model)
  accepted.qpos[:] = qpos[1]
  accepted.qvel[:] = qvel[1]
  mujoco.mj_forward(model, accepted)
  accepted_seed = accepted.qacc.copy()
  model.opt.iterations = configured_iterations

  seeds = np.zeros((3, model.nv), dtype=np.float32)
  seeds[1] = accepted_seed.astype(np.float32)
  seeds[2] = 1000.0
  expected = []
  cold_accepted = None
  for world in range(3):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.qacc_warmstart[:] = seeds[world]
    mujoco.mj_forward(model, data)
    assert data.nefc == dim and data.ncon == 1
    assert data.contact[0].mu == pytest.approx(
        data.contact[0].friction[0] * np.sqrt(data.efc_R[1] / data.efc_R[0]))
    expected.append(np.asarray(data.qacc).copy())
    if world == 1:
      cold_data = mujoco.MjData(model)
      cold_data.qpos[:] = qpos[world]
      cold_data.qvel[:] = qvel[world]
      mujoco.mj_forward(model, cold_data)
      cold_accepted = np.asarray(cold_data.qacc).copy()
  if iterations == 0:
    assert np.linalg.norm(expected[1] - cold_accepted) > 1e-3
    assert np.linalg.norm(expected[2] - seeds[2]) > 1e-3

  sim = MetalSimulation(model, batch_size=3, profile="integrated_euler_v1")
  sim.reset(qpos=qpos, qvel=qvel)
  mj_setState(sim, {"qacc_warmstart": seeds})
  result = sim.assembled_system(recompute=True)
  actual = result["qacc"].detach().cpu().numpy().astype(np.float64)
  np.testing.assert_allclose(actual, np.asarray(expected),
                             rtol=5e-4, atol=5e-3)
  assert torch.equal(result["status"], torch.zeros_like(result["status"]))
  assert torch.isfinite(result["lambda"]).all()


@pytest.mark.parametrize("iterations", [0, 1, 2])
@pytest.mark.parametrize("dim", [3, 4, 6])
def test_pinned_pgs_elliptic_cpu_warmstart_oracle(iterations, dim):
  """Pin CPU accepted/rejected qacc seeds for elliptic PGS budgets."""
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" solver="PGS"
        cone="elliptic" impratio="3.5" iterations="{iterations}"
        tolerance="1e-12"/>
    <worldbody>
      <geom type="plane" size="2 2 .1" condim="{dim}"
          friction=".8 .03 .01"/>
      <body pos="0 0 .04"><freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="{dim}"
            friction=".8 .03 .01"/>
      </body>
    </worldbody>
  </mujoco>""")
  qpos = np.asarray(model.qpos0, dtype=np.float64)
  qvel = np.asarray([.7, -.4, .1, .3, -.2, .5], dtype=np.float64)
  configured_iterations = int(model.opt.iterations)
  model.opt.iterations = max(configured_iterations, 100)
  accepted = mujoco.MjData(model)
  accepted.qpos[:] = qpos
  accepted.qvel[:] = qvel
  mujoco.mj_forward(model, accepted)
  accepted_seed = accepted.qacc.copy()
  model.opt.iterations = configured_iterations

  cold = mujoco.MjData(model)
  cold.qpos[:] = qpos
  cold.qvel[:] = qvel
  mujoco.mj_forward(model, cold)
  warm = mujoco.MjData(model)
  warm.qpos[:] = qpos
  warm.qvel[:] = qvel
  warm.qacc_warmstart[:] = accepted_seed
  mujoco.mj_forward(model, warm)
  rejected_seed = np.full(model.nv, 1000.0)
  rejected = mujoco.MjData(model)
  rejected.qpos[:] = qpos
  rejected.qvel[:] = qvel
  rejected.qacc_warmstart[:] = rejected_seed
  mujoco.mj_forward(model, rejected)
  assert cold.nefc == warm.nefc == rejected.nefc == dim
  assert np.isfinite(warm.qacc).all() and np.isfinite(rejected.qacc).all()
  if iterations == 0:
    assert np.linalg.norm(warm.qacc - cold.qacc) > 1e-3
    assert np.linalg.norm(rejected.qacc - rejected_seed) > 1e-3
