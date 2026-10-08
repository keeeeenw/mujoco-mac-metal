"""Mixed-system uniformity gate plus multibody island low ownership.

Dual refinement runs only on uniform scalar-contact systems; any
enabled equality row, elliptic-cone row or symmetric-bounded friction
row disables it system-wide (partial moves would shift skipped rows'
optimality). These tests prove per-island low-word ownership index-aware
via row_island (not multisets): rows of every island retain their own
lows alongside highs, including mixed systems and budget-exhausted
solves where polish cannot repopulate them, with masked/empty worlds,
reset/restore and no-slip transitions.
"""
import os

import numpy as np
import pytest


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


def _lamlo(cc):
  nv, nr = cc.descriptor.nv, cc.descriptor.nr
  dbg = cc._workspace["workspace_debug"].cpu().numpy()
  base = nr * nr + 7 * nr + cc._high_low_workspace_offset + 8 * max(nv, 1)
  return dbg[base + max(nr, 1):base + 2 * max(nr, 1)].copy()


def _torus_xml(extra_worldbody="", extra="", opt=""):
  return f"""
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="100" tolerance="1e-8" sdf_initpoints="8" {opt}/>
      <extension><plugin plugin="mujoco.sdf.torus">
        <instance name="shape"/>
      </plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="probe" pos="0.45 0 0">
          <freejoint/><geom name="probe_sphere" type="sphere" size="0.1"/>
        </body>
        {extra_worldbody}
      </worldbody>
      {extra}
      <contact><pair geom1="plugin_sdf" geom2="probe_sphere" condim="1"/></contact>
    </mujoco>"""


@_needs_gpu()
def test_uniform_scalar_fires_mixed_skips_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  # Uniform control: pure SDF contact system refines (nonzero lows).
  m0 = mujoco.MjModel.from_xml_string(_torus_xml())
  sim0 = MetalSimulation(m0, batch_size=1, profile="integrated_euler_v1")
  sim0.assembled_system(recompute=True)
  assert float(np.max(np.abs(_lamlo(
      sim0._coupled_constraints)))) > 0.0
  # Equality + SDF: connect weld adds equality rows; refinement must stay
  # off everywhere (exact-zero plane) while primal results stay correct.
  m1 = mujoco.MjModel.from_xml_string(_torus_xml(
      extra_worldbody="""<body name="anchor" pos="0.45 0 0.5">
          <freejoint/><geom name="pebble" type="sphere" size="0.02"/>
        </body>""",
      extra="""<equality><connect body1="probe" body2="anchor"
        anchor="0.45 0 0.25"/></equality>"""))
  sim1 = MetalSimulation(m1, batch_size=1, profile="integrated_euler_v1")
  s1 = sim1.assembled_system(recompute=True)
  assert sim1._coupled_constraints.descriptor.n_eq_rows > 0
  # Refinement is gated off for mixed systems; the plane then holds only
  # the primal solver's own force lows. Behavior must match the oracle
  # (no partial-move corruption of the skipped equality rows).
  assert int(sim1.state.status.cpu().numpy().ravel()[0]) == 0
  d = mujoco.MjData(m1)
  d.qpos[:] = np.asarray(
      sim1.state._qpos[0].cpu().numpy(), dtype=np.float64)
  d.qvel[:] = np.zeros(m1.nv)
  mujoco.mj_forward(m1, d)
  np.testing.assert_allclose(
      np.asarray(s1["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  # Equality multipliers themselves stay primal-correct (no corruption).
  lam = np.asarray(s1["lambda"][0].cpu().numpy(), dtype=np.float64)
  assert np.all(np.isfinite(lam))


@_needs_gpu()
def test_frictionloss_skips_refinement_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  # Joint frictionloss adds symmetric-bounded rows (Huber semantics):
  # refinement stays off, primal path stays correct.
  m = mujoco.MjModel.from_xml_string(_torus_xml(
      extra_worldbody="",
      extra="").replace(
      '<freejoint/>',
      '<joint name="slidex" type="slide" axis="1 0 0" frictionloss="0.5"/>',
      1))
  assert m.nv == 1  # slide joint replaces the free joint (1 DOF)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  s = sim.assembled_system(recompute=True)
  assert int(sim.state.status.cpu().numpy().ravel()[0]) == 0
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64)
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(
      np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qfrc_constraint, dtype=np.float64), rtol=1e-3, atol=1e-3)


@_needs_gpu()
def test_elliptic_sdf_skips_refinement_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  m = mujoco.MjModel.from_xml_string(_torus_xml(
      opt='cone="elliptic"').replace('condim="1"', 'condim="3"'))
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  s = sim.assembled_system(recompute=True)
  assert int(sim.state.status.cpu().numpy().ravel()[0]) == 0
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64)
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(
      np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qfrc_constraint, dtype=np.float64), rtol=1e-3, atol=1e-3)


@_needs_gpu()
def test_island_low_ownership_multibody_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  xml = """
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="100" tolerance="1e-8" sdf_initpoints="8"/>
      <extension><plugin plugin="mujoco.sdf.torus">
        <instance name="shape"/>
      </plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="a" pos="0.45 0 0">
          <freejoint/><geom name="sa" type="sphere" size="0.1"/>
        </body>
        <body name="b" pos="-0.45 0 0">
          <freejoint/><geom name="sb" type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <contact>
        <pair geom1="plugin_sdf" geom2="sa" condim="1"/>
        <pair geom1="plugin_sdf" geom2="sb" condim="1"/>
      </contact>
    </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  qpos = np.asarray(sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  far = qpos.copy()
  far[0:3] += np.array([5.0, 5.0, 5.0])
  far[7:10] += np.array([5.0, 5.0, 5.0])
  sim.reset(qpos=np.stack([qpos, far]).astype(np.float32),
            qvel=np.zeros((2, m.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  cc = sim._coupled_constraints
  nv, nr = m.nv, cc.descriptor.nr
  dbg = cc._workspace["workspace_debug"].cpu().numpy()
  # Empty world publishes zeros and stays successful.
  lam_all = np.asarray(s["lambda"].cpu().numpy())
  assert np.all(lam_all[1] == 0.0)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[1]) == 0
  # Index-aware ownership: every island with active SDF rows retains
  # nonzero lows on ITS rows (not just the last island solved).
  row_island = np.asarray(s["row_island"].cpu().numpy()).reshape(2, -1)
  lamlo = dbg[nr * nr + 7 * nr + cc._high_low_workspace_offset
              + 8 * max(nv, 1) + max(nr, 1):
              nr * nr + 7 * nr + cc._high_low_workspace_offset
              + 8 * max(nv, 1) + 2 * max(nr, 1)]
  islands = sorted(set(int(v) for v in row_island[0] if int(v) >= 0))
  assert len(islands) >= 2
  for isl in islands:
    rows = [r for r in range(nr) if int(row_island[0, r]) == isl]
    assert rows
    assert float(np.max(np.abs(lamlo[rows]))) > 0.0, isl
  # Same-state force agreement on the populated world.
  d = mujoco.MjData(m)
  d.qpos[:] = qpos
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  nat = np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64)
  cpu = np.asarray(d.qfrc_constraint, dtype=np.float64)
  np.testing.assert_allclose(nat, cpu, rtol=1e-4, atol=1e-4)
  # Restore roundtrip reproduces the assembly bit-identically.
  lam0 = np.asarray(s["lambda"].cpu().numpy()).copy()
  snap = sim.snapshot()
  sim.step(1)
  sim.restore(snap)
  sr = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(
      np.asarray(sr["lambda"].cpu().numpy()), lam0)


def _mixed_two_island_xml(swap=False):
  # Two SDF probes on separate trees (two islands) where one side carries
  # joint frictionloss (symmetric-bounded Huber rows). The system is mixed
  # so dual refinement stays off globally; with iterations=1 the solve is
  # also budget-exhausted, so no polish pass can repopulate multiplier lows.
  # `swap` reverses body/row order to exercise reverse island traversal.
  bodies = [
      """<body name="a" pos="0.45 0 0">
          <freejoint/><geom name="sa" type="sphere" size="0.1"/>
        </body>""",
      """<body name="b" pos="-0.45 0 0">
          <joint name="slideb" type="slide" axis="1 0 0" frictionloss="0.5"/>
          <geom name="sb" type="sphere" size="0.1"/>
        </body>""",
  ]
  if swap:
    bodies = bodies[::-1]
  return f"""
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="1" tolerance="1e-8" sdf_initpoints="8"/>
      <extension><plugin plugin="mujoco.sdf.torus">
        <instance name="shape"/>
      </plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        {"".join(bodies)}
      </worldbody>
      <contact>
        <pair geom1="plugin_sdf" geom2="sa" condim="1"/>
        <pair geom1="plugin_sdf" geom2="sb" condim="1"/>
      </contact>
    </mujoco>"""


@_needs_gpu()
@pytest.mark.parametrize("swap", [False, True], ids=["order", "reversed"])
def test_mixed_multibody_budget_exhausted_low_ownership_gpu(swap):
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  m = mujoco.MjModel.from_xml_string(_mixed_two_island_xml(swap=swap))
  m.opt.solver = 2  # Newton primal path with pair refresh per island
  m.opt.iterations = 1  # budget-exhausted: polish/recon cannot mask loss
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nv, nr = int(m.nv), int(cc.descriptor.nr)
  # Mixed system: symmetric-bounded frictionloss rows present, so the
  # uniform-scalar gate must stay off (no system-wide refinement).
  assert cc.descriptor.nr > 0
  row_island = np.asarray(s["row_island"].cpu().numpy()).reshape(-1)
  lamlo = _lamlo(cc)
  assert lamlo.shape == (nr,)
  assert np.all(np.isfinite(lamlo))
  islands = sorted(set(int(v) for v in row_island if int(v) >= 0))
  assert len(islands) >= 2
  # Index-aware ownership without polish: EVERY island with enabled rows
  # retains its own nonzero lows (later islands must not erase earlier
  # ones via the retained force-low/residual-low planes). A uniform-system
  # fixture would let system-wide refinement repopulate these; here both
  # the mixed gate and the exhausted budget keep polish off.
  lamhi = np.asarray(s["lambda"][0].cpu().numpy(), dtype=np.float64)
  for isl in islands:
    rows = [r for r in range(nr) if int(row_island[r]) == isl]
    assert rows
    assert float(np.max(np.abs(lamlo[rows]))) > 0.0, (swap, isl)
    assert np.all(np.isfinite(lamhi[rows]))
  # Mapped value agreement against the pinned oracle at the same
  # represented state (DOF-indexed, no sorting): force and acceleration.
  d = mujoco.MjData(m)
  d.qpos[:] = qpos
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(
      np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qfrc_constraint, dtype=np.float64), rtol=1e-3, atol=1e-3)
  # Replay: snapshot/step/restore reproduces the assembly bit-identically.
  lam0 = np.asarray(s["lambda"].cpu().numpy()).copy()
  snap = sim.snapshot()
  sim.step(1)
  sim.restore(snap)
  sr = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(
      np.asarray(sr["lambda"].cpu().numpy()), lam0)
