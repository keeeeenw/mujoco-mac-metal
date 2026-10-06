import os

import numpy as np
import pytest

from mujoco_metal.constraint_impedance import (
    ConeGroup,
    compile_contact_cone_groups,
    exact_constraint_diagonal_cpu,
    recompute_impedance_cpu,
)


def test_static_contact_cone_lowering_preserves_canonical_rigid_and_flex_rows():
  from types import SimpleNamespace
  import mujoco

  elliptic = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
  pyramidal = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
  descriptor = SimpleNamespace(
      contact_condim_packed=np.asarray([
          1, 0, elliptic,
          3, 1, elliptic,
          4, 4, pyramidal,
      ], dtype=np.int32),
      pair_contact_offset=np.asarray([0, 1, 3], dtype=np.int32),
      contact_friction=np.asarray([
          [.7, .7, .4, .2, .2],
          [.5, .5, .3, .1, .1],
          [.5, .5, .3, .1, .1],
      ], dtype=np.float32),
      npairs=2, nr_joint=7, flex_contact_base=15,
      flex_contact_descriptor=SimpleNamespace(
          row_start=np.asarray([0, 6], dtype=np.int32),
          condim=np.asarray([3, 6], dtype=np.int32),
          cone=np.asarray([elliptic, pyramidal], dtype=np.int32),
          friction=np.asarray([
              [.8, .8, .4, .2, .2], [.6, .6, .3, .1, .1],
          ], dtype=np.float32)))
  groups, friction = compile_contact_cone_groups(descriptor)
  np.testing.assert_array_equal(groups, [
      [8, 3, 1], [11, 4, 2], [15, 3, 1], [21, 6, 2],
  ])
  np.testing.assert_allclose(friction, [
      [.5, .5, .3, .1, .1], [.5, .5, .3, .1, .1],
      [.8, .8, .4, .2, .2], [.6, .6, .3, .1, .1],
  ])


def test_static_contact_cone_lowering_returns_one_inert_empty_sentinel():
  from types import SimpleNamespace
  import mujoco

  groups, friction = compile_contact_cone_groups(SimpleNamespace(
      contact_condim_packed=np.zeros((0,), dtype=np.int32),
      pair_contact_offset=np.asarray([0], dtype=np.int32),
      contact_friction=np.zeros((0, 5), dtype=np.float32),
      npairs=0, nr_joint=0, flex_contact_base=0,
      flex_contact_descriptor=None))
  np.testing.assert_array_equal(groups, np.zeros((1, 3), dtype=np.int32))
  np.testing.assert_array_equal(friction, np.zeros((1, 5), dtype=np.float32))


def test_static_cone_lowering_matches_compiled_rigid_contact_descriptor():
  import mujoco
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body pos="0 0 .09"><freejoint/>
        <geom type="box" size=".1 .08 .1" mass="1" condim="4"
              friction=".8 .4 .2"/>
      </body>
    </worldbody>
  </mujoco>""")
  descriptor = lower_coupled_constraints(model)
  groups, friction = compile_contact_cone_groups(descriptor)
  packed = descriptor.contact_condim_packed.reshape(-1, 3)
  expected = []
  for slot, (condim, offset, cone) in enumerate(packed):
    if int(condim) == 1:
      continue
    expected.append((descriptor.nr_joint + int(offset), int(condim),
                     1 if int(cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
                     else 2))
    np.testing.assert_allclose(friction[len(expected) - 1],
                               descriptor.contact_friction[slot])
  np.testing.assert_array_equal(groups, np.asarray(expected, dtype=np.int32))


def test_impedance_metadata_preflight_rejects_bad_static_cone_rows_without_torch():
  from mujoco_metal.constraint_impedance import MetalConstraintImpedance

  with pytest.raises(ValueError, match=r"cone_groups.*\[G,3\]"):
    MetalConstraintImpedance(
        batch_size=1, row_capacity=4, dof_capacity=2,
        cone_groups=np.asarray([[0.0, 3.0, 1.0]]),
        friction=np.ones((1, 5)))
  with pytest.raises(ValueError, match="invalid row span"):
    MetalConstraintImpedance(
        batch_size=1, row_capacity=4, dof_capacity=2,
        cone_groups=np.asarray([[2, 3, 1]], dtype=np.int32),
        friction=np.ones((1, 5)))
  with pytest.raises(ValueError, match="invalid row span"):
    MetalConstraintImpedance(
        batch_size=1, row_capacity=8, dof_capacity=2,
        cone_groups=np.asarray([[2**32, 3, 1]], dtype=np.int64),
        friction=np.ones((1, 5)))
  with pytest.raises(ValueError, match="must not overlap"):
    MetalConstraintImpedance(
        batch_size=1, row_capacity=8, dof_capacity=2,
        cone_groups=np.asarray([[0, 3, 1], [2, 3, 1]], dtype=np.int32),
        friction=np.ones((2, 5)))


def test_exact_impedance_uses_pinned_minimum_effective_contact_friction():
  diagonal = np.asarray([[1.0, 1.0, 1.0]])
  impedance = np.full_like(diagonal, 0.5)
  groups = (ConeGroup(0, 3, "elliptic", (0.0, 0.0, 0.0, 0.0, 0.0)),)
  R, _, mu = recompute_impedance_cpu(
      diagonal, impedance, cone_groups=groups, impratio=1.0)
  np.testing.assert_allclose(mu[0, 0], 1.0e-5, rtol=0, atol=1e-12)
  np.testing.assert_allclose(R[0, 2], R[0, 1], rtol=0, atol=1e-12)


def test_component_rhs_contraction_fixture_is_batched_and_row_major():
  batch, nr, nv = 2, 8, 2
  J = np.arange(batch * nr * nv, dtype=np.float32).reshape(batch, nr, nv)
  J = (J % 7 - 2) * 0.125
  inv_mass_diagonal = np.asarray([[0.5, 0.25], [0.2, 0.4]], dtype=np.float32)
  inverse_rows = J * inv_mass_diagonal[:, None, :]
  padded = np.zeros((batch, nr + 1, nv), dtype=np.float32)
  padded[:, :nr] = inverse_rows
  contracted = np.sum(J * padded[:, :nr], axis=-1)
  independent = np.zeros((batch, nr), dtype=np.float32)
  for world in range(batch):
    for row in range(nr):
      independent[world, row] = sum(
          J[world, row, dof] * J[world, row, dof]
          * inv_mass_diagonal[world, dof] for dof in range(nv))
  np.testing.assert_allclose(contracted, independent, rtol=1e-7, atol=1e-8)
  assert padded.shape == (batch, nr + 1, nv)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in MPS kernel qualification")
def test_native_exact_diagonal_and_cone_impedance_component_rhs():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("MPS is unavailable")
  from mujoco_metal.constraint_impedance import MetalConstraintImpedance

  batch, nr, nv = 2, 8, 2
  J_np = np.arange(batch * nr * nv, dtype=np.float32).reshape(batch, nr, nv)
  J_np = (J_np % 7 - 2) * 0.125
  inverse_np = J_np * np.asarray([[[0.5, 0.25]], [[0.2, 0.4]]],
                                  dtype=np.float32)
  # The component solve contract pads one final RHS row for qfrc_smooth.
  inverse_padded = np.zeros((batch, nr + 1, nv), dtype=np.float32)
  inverse_padded[:, :nr] = inverse_np
  active = np.asarray([
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1, 1, 1, 1],
  ], dtype=np.int32)
  impedance = np.asarray([
      [0.5, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6],
      [0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7],
  ], dtype=np.float32)
  groups = np.asarray([[1, 3, 1], [4, 3, 2]], dtype=np.int32)
  friction = np.asarray([[0.8, 0.8, 0.4, 0.2, 0.2],
                         [0.6, 0.6, 0.3, 0.1, 0.1]], dtype=np.float32)
  program = MetalConstraintImpedance(
      batch_size=batch, row_capacity=nr, dof_capacity=nv,
      cone_groups=groups, friction=friction)
  device = torch.device("mps")
  impedance_storage = np.zeros((batch, nr + 1), dtype=np.float32)
  impedance_storage[:, :nr] = impedance
  impedance_view = torch.as_tensor(impedance_storage, device=device)[:, :nr]
  assert not impedance_view.is_contiguous()
  result = program.run_device(
      torch.as_tensor(J_np, device=device),
      torch.as_tensor(inverse_padded, device=device),
      impedance_view,
      active_rows=torch.as_tensor(active, device=device),
      impratio=2.0)
  diag_cpu = np.sum(J_np * inverse_np, axis=-1)
  expected_diag = np.where(active != 0, diag_cpu, 0.0)
  expected_R = np.zeros((batch, nr), dtype=np.float64)
  expected_diagA = np.zeros_like(expected_R)
  expected_mu = np.zeros((batch, 2), dtype=np.float64)
  for world in range(batch):
    groups_for_world = [
        ConeGroup(1, 3, "elliptic", tuple(friction[0])),
        ConeGroup(4, 3, "pyramidal", tuple(friction[1])),
    ]
    if active[world, 1] == 0:
      groups_for_world.pop(0)
    R, diagA, mu = recompute_impedance_cpu(
        expected_diag[world:world + 1], impedance[world:world + 1],
        cone_groups=groups_for_world, impratio=2.0)
    expected_R[world], expected_diagA[world] = R[0], diagA[0]
    expected_R[world, active[world] == 0] = 0.0
    expected_diagA[world, active[world] == 0] = 0.0
    if active[world, 1]:
      expected_mu[world, 0] = mu[0, 0]
      expected_mu[world, 1] = mu[0, 1]
    else:
      expected_mu[world, 1] = mu[0, 0]
  np.testing.assert_allclose(result["diagonal"].cpu().numpy(), expected_diag,
                             rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(result["R"].cpu().numpy(), expected_R,
                             rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(result["diagA"].cpu().numpy(), expected_diagA,
                             rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(result["contact_mu"].cpu().numpy(), expected_mu,
                             rtol=2e-5, atol=2e-6)


def test_exact_constraint_diagonal_matches_independent_principal_solves():
  jacobian = np.asarray([
      [[1.0, 2.0, 0.5], [0.0, -1.0, 3.0], [2.0, 0.0, 1.0]],
      [[-2.0, 1.0, 0.0], [0.5, 1.5, -1.0], [1.0, 1.0, 1.0]],
  ])
  mass = np.asarray([
      [[2.0, 0.2, 0.0], [0.2, 3.0, 0.1], [0.0, 0.1, 4.0]],
      [[5.0, 0.0, 0.0], [0.0, 2.0, 0.3], [0.0, 0.3, 3.0]],
  ])
  active_dof = np.asarray([[1, 1, 1], [1, 0, 1]], dtype=bool)
  active_rows = np.asarray([[1, 1, 1], [1, 0, 1]], dtype=bool)
  got = exact_constraint_diagonal_cpu(
      jacobian, mass, active_dof=active_dof, active_rows=active_rows)
  expected = np.zeros((2, 3))
  for world in range(2):
    dofs = np.flatnonzero(active_dof[world])
    for row in np.flatnonzero(active_rows[world]):
      j = jacobian[world, row, dofs]
      expected[world, row] = j @ np.linalg.solve(
          mass[world][np.ix_(dofs, dofs)], j)
  np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)
  assert got[1, 1] == 0.0
  assert not np.isclose(got[0, 0], got[1, 0])


def test_dense_pgs_acceleration_pair_is_residual_correction_of_mass_equation():
  """The published low word must close the same force/mass equation as qacc."""
  mass64 = np.asarray([
      [1.7, -0.01879811, 0.0031],
      [-0.01879811, 0.02491484, -0.0042],
      [0.0031, -0.0042, 0.79999995],
  ], dtype=np.float64)
  mass32 = mass64.astype(np.float32)
  lower32 = np.linalg.cholesky(mass64).astype(np.float32)
  smooth_force = np.asarray([0.009297324, 0.0, -0.13], dtype=np.float32)
  jacobian = np.asarray([[0.0, 1.0, -1.0], [0.2, -0.3, 0.0]],
                        dtype=np.float32)
  multipliers = np.asarray([-9.234867, 0.125], dtype=np.float32)

  # Reproduce the rounded legacy construction independently: float32 smooth
  # and M^-1 J^T solves followed by float32 accumulation.
  q0 = np.linalg.solve(lower32, smooth_force).astype(np.float32)
  q0 = np.linalg.solve(lower32.T, q0).astype(np.float32)
  columns = []
  for row in jacobian:
    y = np.linalg.solve(lower32, row).astype(np.float32)
    columns.append(np.linalg.solve(lower32.T, y).astype(np.float32))
  approximate = q0.copy()
  for row, solved in zip(multipliers, columns):
    approximate = (approximate + solved * row).astype(np.float32)

  # The accepted RHS is formed from the same rows and multipliers. Solve the
  # residual against the original M, rather than treating the rounded factor
  # solve as exact. This independent float64 oracle defines qacc_low's meaning.
  total_force = smooth_force.astype(np.float64) + jacobian.astype(np.float64).T @ multipliers.astype(np.float64)
  residual = total_force - mass64 @ approximate.astype(np.float64)
  correction = np.linalg.solve(mass64, residual)
  paired = approximate.astype(np.float64) + correction
  legacy_error = np.linalg.norm(
      total_force - mass64 @ approximate.astype(np.float64))
  paired_error = np.linalg.norm(total_force - mass64 @ paired)
  np.testing.assert_allclose(paired, np.linalg.solve(mass64, total_force),
                             rtol=2e-7, atol=2e-8)
  assert paired_error < legacy_error * 1e-8
  assert np.any(paired.astype(np.float32) != approximate)


def test_exact_impedance_updates_scalar_elliptic_and_pyramid_rows_per_world():
  diagonal = np.asarray([[2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.5, 2.5],
                         [5.0, 2.0, 6.0, 4.0, 1.0, 3.0, 2.5, 4.0]])
  impedance = np.asarray([[0.2, 0.4, 0.6, 0.25, 0.5, 0.75, 0.45, 0.55],
                          [0.3, 0.5, 0.7, 0.35, 0.55, 0.65, 0.4, 0.6]])
  groups = (
      ConeGroup(1, 3, "elliptic", (0.8, 0.4, 0.2, 0.0, 0.0)),
      ConeGroup(4, 3, "pyramidal", (0.6, 0.3, 0.2, 0.0, 0.0)),
  )
  R, diagA, mu = recompute_impedance_cpu(
      diagonal, impedance, cone_groups=groups, impratio=2.0)
  base = np.maximum(1e-15, (1.0 - impedance) * diagonal / impedance)
  np.testing.assert_allclose(R[:, 0], base[:, 0])  # scalar row
  np.testing.assert_allclose(R[:, 1], base[:, 1])  # contact normal
  np.testing.assert_allclose(R[:, 2], base[:, 1] / 2.0)
  np.testing.assert_allclose(
      R[:, 4], 2.0 * (0.6 / np.sqrt(2.0)) ** 2 * base[:, 4])
  np.testing.assert_allclose(R[:, 5], R[:, 4])
  np.testing.assert_allclose(R[:, 6], R[:, 4])
  np.testing.assert_allclose(R[:, 7], R[:, 4])
  np.testing.assert_allclose(mu[:, 0], 0.8 / np.sqrt(2.0))
  np.testing.assert_allclose(mu[:, 1], 0.6 / np.sqrt(2.0))
  np.testing.assert_allclose(diagA, R * impedance / (1.0 - impedance))
  assert not np.isclose(R[0, 1], R[1, 1])


def test_exact_impedance_rejects_invalid_dimensions_and_impedance():
  with pytest.raises(ValueError, match=r"share \[batch,nr\]"):
    recompute_impedance_cpu(np.ones((2, 3)), np.ones((2, 2)) * 0.5)
  with pytest.raises(ValueError, match="strictly between"):
    recompute_impedance_cpu(np.ones((1, 2)), np.asarray([[0.5, 1.0]]))
  with pytest.raises(ValueError, match="cone rows"):
    recompute_impedance_cpu(
        np.ones((1, 3)), np.ones((1, 3)) * 0.5,
        cone_groups=(ConeGroup(1, 3, "elliptic", (1, 1, 1, 0, 0)),))


def test_preassembled_rows_flag_is_independent_of_cached_velocity_epoch():
  from mujoco_metal.coupled_constraints import _preassembled_rows_stage

  dims = np.zeros(8, dtype=np.int32)
  dims[7] = 2  # cached-velocity epoch semantics
  with _preassembled_rows_stage(dims):
    assert int(dims[7]) == (2 | 8)
  assert int(dims[7]) == 2


def test_pinned_mujoco_diagexact_matches_chain_joint_equality_oracle():
  """The flag replaces the approximate equality diagonal with J M^-1 J.T."""
  import mujoco

  xml = """<mujoco>
    <worldbody>
      <body>
        <joint name="a" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".05 .5" fromto="0 0 0 1 0 0" mass="1"/>
        <body pos="1 0 0">
          <joint name="b" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size=".04 .5" fromto="0 0 0 1 0 0" mass="1"/>
        </body>
      </body>
    </worldbody>
    <equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
  </mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  enable_bit = int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  results = []
  for exact in (False, True):
    model.opt.enableflags = ((int(model.opt.enableflags) & ~enable_bit)
                             | (enable_bit if exact else 0))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.3, -0.2]
    mujoco.mj_forward(model, data)
    full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, full_mass)
    J = np.asarray(data.efc_J, dtype=np.float64).reshape(data.nefc, model.nv)
    exact_diag = np.asarray([row @ np.linalg.solve(full_mass, row) for row in J])
    results.append((np.asarray(data.efc_diagA).copy(),
                    np.asarray(data.efc_R).copy(), exact_diag))
  approximate, exact = results
  np.testing.assert_allclose(exact[0], exact[2], rtol=2e-13, atol=2e-13)
  assert abs(float(approximate[0][0] - exact[0][0])) > 1.0
  imp = float(model.jnt_solimp[0, 1])
  np.testing.assert_allclose(
      exact[1][0], (1.0 - imp) / imp * exact[2][0],
      rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
def test_pinned_mujoco_contact_cones_match_exact_impedance_oracle(cone):
  """Verify exact contact R, adjusted diagA and mu for both cone models."""
  import mujoco

  xml = f"""<mujoco><option cone="{cone}" impratio="2"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .09"><freejoint/>
        <geom type="box" size=".1 .08 .1" mass="1"
              friction=".8 .4 .2"/>
      </body>
    </worldbody>
  </mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 4 and data.nefc > 1
  full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, full_mass)
  jacobian = np.asarray(data.efc_J, dtype=np.float64).reshape(data.nefc, model.nv)
  exact_diag = np.asarray([
      row @ np.linalg.solve(full_mass, row) for row in jacobian])
  impedance = np.zeros(data.nefc, dtype=np.float64)
  groups = []
  for contact_index in range(data.ncon):
    contact = data.contact[contact_index]
    rows = (contact.dim if cone == "elliptic"
            else 2 * (contact.dim - 1))
    start = int(contact.efc_address)
    # The contact is penetrated beyond solimp[2], so getimpedance selects d1.
    impedance[start:start + rows] = float(contact.solimp[1])
    groups.append(ConeGroup(
        start, int(contact.dim), cone,
        tuple(np.asarray(contact.friction, dtype=np.float64))))
  expected_R, expected_diagA, expected_mu = recompute_impedance_cpu(
      exact_diag[None, :], impedance[None, :], cone_groups=groups,
      impratio=float(model.opt.impratio))
  np.testing.assert_allclose(data.efc_R, expected_R[0], rtol=3e-13, atol=3e-13)
  np.testing.assert_allclose(data.efc_diagA, expected_diagA[0],
                             rtol=3e-13, atol=3e-13)
  np.testing.assert_allclose(
      [data.contact[i].mu for i in range(data.ncon)], expected_mu[0],
      rtol=3e-13, atol=3e-13)


def test_final_exact_status_merge_survives_reused_mass_status_buffer():
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import _merge_exact_world_status

  # The exact mass solve exposes a borrowed component status tensor. A later
  # successful RHS solve can clear/reuse it, while the helper's retained
  # per-world status must continue to report the earlier mass failure.
  borrowed_mass_status = torch.tensor([1, 0], dtype=torch.int32)
  retained_world_status = borrowed_mass_status.clone()
  exact_data = {"mass_status": borrowed_mass_status,
                "status": retained_world_status}
  borrowed_mass_status.zero_()
  out_status = torch.zeros((2,), dtype=torch.int32)
  _merge_exact_world_status(torch, out_status, exact_data)
  assert out_status.tolist() == [1, 0]
  assert exact_data["mass_status"].tolist() == [0, 0]


def test_component_exact_rows_run_before_shared_rhs_solution_overwrite():
  from mujoco_metal.coupled_constraints import (
      _run_component_solve_after_prepare,
  )

  # Both operations borrow a component solver's persistent output storage. The
  # exact diagonal helper must consume its mass-row solve first and retain R in
  # its own output before the smooth/contact RHS solve replaces that storage.
  shared = [0.0, 0.0, 0.0]
  calls = []

  def prepare_exact_rows():
    shared[:] = [0.25, 0.5, 0.75]
    calls.append("exact")
    return {"R": tuple(shared)}

  def solve_smooth_rhs():
    assert calls == ["exact"]
    assert shared == [0.25, 0.5, 0.75]
    shared[:] = [10.0, 20.0, 30.0]
    calls.append("smooth")
    return shared, [0, 0]

  exact, (solution, status) = _run_component_solve_after_prepare(
      prepare_exact_rows, solve_smooth_rhs)
  assert calls == ["exact", "smooth"]
  assert exact["R"] == (0.25, 0.5, 0.75)
  assert solution is shared and solution == [10.0, 20.0, 30.0]
  assert status == [0, 0]
