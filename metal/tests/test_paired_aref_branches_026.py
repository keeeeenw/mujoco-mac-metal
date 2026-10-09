"""Paired-aref branch coverage: every changed solref/solimp path.

Each variant exercises different low-kernel branches (positive vs direct
solref, refsafe clamping, override values, impedance lerp vs pow,
midpoint/width/margin, pyramidal/elliptic cones, condim 1/3, mixed
analytic+SDF, batch>1, moving states) and asserts the paired reference
E (hi+lo in float64) against the pinned CPU oracle. Static states use a
tight tier (velocity term vanishes); the moving case uses a documented
qvel-scaled tier (float32 qvel input rounding amplified by B).
"""
import os

import numpy as np
import pytest


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


def _torus_xml(**kw):
  opt = kw.get("opt", "")
  pair_extra = kw.get("pair_extra", "")
  condim = kw.get("condim", "1")
  sphere = kw.get("sphere",
                  '<geom name="probe_sphere" type="sphere" size="0.1"/>')
  extra_pair = kw.get("extra_pair", "")
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
          <freejoint/>{sphere}</body>
      </worldbody>
      <contact><pair geom1="plugin_sdf" geom2="probe_sphere"
        condim="{condim}" {pair_extra}/>{extra_pair}</contact>
    </mujoco>"""


def _check_variant(xml, qpos=None, qvel=None, batch=1, etier=5e-9,
                   ftier=1e-6):
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, batch_size=batch,
                        profile="integrated_euler_v1")
  if qpos is None:
    qpos = np.asarray(
        sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  if qvel is None:
    qvel = np.zeros(model.nv)
  # Same represented state on both sides: the device consumes float32, so
  # the CPU oracle runs at float32(qpos/qvel) reinterpreted as float64.
  qpos32 = qpos.astype(np.float32).astype(np.float64)
  qvel32 = qvel.astype(np.float32).astype(np.float64)
  sim.reset(qpos=np.tile(qpos.astype(np.float32), (batch, 1)),
            qvel=np.tile(qvel.astype(np.float32), (batch, 1)))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nslots = int(cc.descriptor.ncontacts_max)
  d = mujoco.MjData(model)
  d.qpos[:] = qpos32
  d.qvel[:] = qvel32
  mujoco.mj_forward(model, d)
  _aref = np.asarray(d.efc_aref, dtype=np.float64)
  cpu = [(float(d.contact[c].dist), np.asarray(d.contact[c].pos),
          float(_aref[int(d.contact[c].efc_address)]),
          int(d.contact[c].efc_address)) for c in range(d.ncon)]
  assert len(cpu) == d.ncon and d.ncon > 0
  rdb = cc._workspace["contact_row_data"].cpu().numpy().reshape(
      batch, nslots, 60)[:, :, :42].reshape(batch, nslots, 6, 7)
  rec = cc._workspace["contact_pair_records"].cpu().numpy().reshape(
      batch, nslots, 37)
  worst = 0.0
  worst_sdf = 0.0
  for w in range(batch):
    npos = np.asarray(s["contact_position"][w].cpu().numpy())
    mask = np.asarray(s["contact_mask"][w].cpu().numpy())
    used = set()
    for i in [x for x in range(nslots) if mask[x] > 0.5]:
      c = min((x for x in range(d.ncon) if x not in used),
              key=lambda x: float(np.linalg.norm(cpu[x][1] - npos[i])))
      used.add(c)
      assert float(np.linalg.norm(cpu[c][1] - npos[i])) < 1e-4
      E = float(rdb[w, i, 0, 3]) + float(rdb[w, i, 0, 6])
      # Analytic records carry no distance low word (exact 0.0): their
      # float32 distance input bounds E at K-input-rounding scale (~1e-5).
      # SDF slots with a live dist correction hold the tight tier.
      tier = etier if float(rec[w, i, 9]) != 0.0 else max(etier, 1e-5)
      assert abs(E - cpu[c][2]) < tier, (i, E - cpu[c][2])
      worst = max(worst, abs(E - cpu[c][2]))
      if float(rec[w, i, 9]) != 0.0:
        worst_sdf = max(worst_sdf, abs(E - cpu[c][2]))
  assert worst < max(etier, 1e-5), worst
  assert worst_sdf < etier, worst_sdf
  # Mapped publication per world (no sorting; DOF-indexed): every world
  # agrees with the same-state oracle.
  for w in range(batch):
    nat = np.asarray(s["qfrc_constraint"][w].cpu().numpy(), dtype=np.float64)
    cpu_f = np.asarray(d.qfrc_constraint, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(cpu_f))))
    assert float(np.max(np.abs(nat - cpu_f))) / scale < ftier, w
  return worst


def _check_contact_transition(xml, etier):
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  qpos = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  far = qpos.copy()
  far[0:3] += np.array([5.0, 5.0, 5.0])
  sim.reset(qpos=np.stack(
      [qpos.astype(np.float32), far.astype(np.float32)]),
      qvel=np.zeros((2, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  status = np.asarray(sim.state.status.cpu().numpy()).ravel()
  assert int(status[0]) == 0 and int(status[1]) == 0
  mask = np.asarray(s["contact_mask"].cpu().numpy())
  assert float(np.sum(mask[0])) > 0.0
  assert float(np.sum(mask[1])) == 0.0
  # Contact world matches the same-state oracle; empty world publishes
  # zero force.
  d = mujoco.MjData(model)
  d.qpos[:] = qpos.astype(np.float32).astype(np.float64)
  d.qvel[:] = np.zeros(model.nv)
  mujoco.mj_forward(model, d)
  assert d.ncon > 0
  nat = np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64)
  cpu_f = np.asarray(d.qfrc_constraint, dtype=np.float64)
  scale = max(1.0, float(np.max(np.abs(cpu_f))))
  assert float(np.max(np.abs(nat - cpu_f))) / scale < 1e-6
  assert float(np.max(np.abs(
      np.asarray(s["qfrc_constraint"][1].cpu().numpy())))) == 0.0


@_needs_gpu()
@pytest.mark.parametrize("variant", [
    "direct-solref",
    "refsafe-clamp",
    "power-1-lerp",
    "power-2-pow",
    "margin-gap",
    "condim-3-pyramidal",
    "condim-4-pyramidal",
    "condim-3-elliptic",
    "condim-3-elliptic-solreffriction",
    "geom-mixing",
    "mixed-analytic-sdf",
    "moving-state",
    "batch-2",
    "contact-transition",
    "clamp-equal-d0d1",
    "clamp-low-bound",
])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_paired_aref_branch_variants_gpu(variant):
  import mujoco
  flag = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
  if variant == "direct-solref":
    m = mujoco.MjModel.from_xml_string(_torus_xml())
    m.opt.enableflags |= flag
    m.opt.o_solref[:] = (-0.01, -0.02)
    _check_variant_xml(m, 5e-9)
  elif variant == "refsafe-clamp":
    m = mujoco.MjModel.from_xml_string(_torus_xml())
    m.opt.enableflags |= flag
    m.opt.o_solref[:] = (0.001, 1.0)
    _check_variant_xml(m, 5e-9)
  elif variant in ("power-1-lerp", "power-2-pow"):
    m = mujoco.MjModel.from_xml_string(_torus_xml())
    m.opt.enableflags |= flag
    p = 1.0 if variant == "power-1-lerp" else 2.0
    m.opt.o_solimp[:] = (0.9, 0.95, 0.001, 0.5, p)
    # Shallow states amplify descent dist differences (K ~ 1e-3/width):
    # measured dist stalls ~2e-9 explain aref gaps ~5e-6, far below any
    # formula-level branch error. Tier documents the input-noise bound.
    _check_shallow(m, 1e-5)
  elif variant == "margin-gap":
    # Override margin plus an explicit pair gap: gap shifts admission while
    # margin enters aref. Both must match the oracle with active contacts.
    m = mujoco.MjModel.from_xml_string(
        _torus_xml(pair_extra='gap="0.001"'))
    m.opt.enableflags |= flag
    m.opt.o_margin = 0.002
    _check_variant_xml(m, 5e-9)
  elif variant == "condim-3-pyramidal":
    # Friction-block solver tolerance scale for force (normal-row E stays
    # exact, proving the pairing; the force residual is solver tolerance).
    _check_variant(_torus_xml(condim="3"), etier=5e-9, ftier=5e-6)
  elif variant == "condim-4-pyramidal":
    # Larger friction blocks accumulate solver residual (normal-row E stays
    # exact, proving the pairing); the force tier scales with block size.
    _check_variant(_torus_xml(condim="4"), etier=5e-9, ftier=1e-5)
  elif variant == "condim-3-elliptic-solreffriction":
    # Elliptic cone with nonzero override solreffriction exercises the
    # friction_B branch (fr1 > 0 selects 2/(d1*fr0)); normal-row E plus
    # mapped force prove the pairing through the tangent path.
    m = mujoco.MjModel.from_xml_string(
        _torus_xml(condim="3", opt='cone="elliptic"'))
    m.opt.enableflags |= flag
    m.opt.o_solref[:] = (0.03, 1.5)
    _check_variant_xml(m, 5e-9, ftier=5e-6)
  elif variant == "condim-3-elliptic":
    # Cone-block solver tolerance scale for force (normal-row E stays
    # exact, proving the pairing; the force residual is solver tolerance).
    _check_variant(_torus_xml(condim="3", opt='cone="elliptic"'),
                   etier=5e-9, ftier=5e-6)
  elif variant == "geom-mixing":
    # Generated (dynamic) geom pair WITHOUT an explicit <pair> block, so the
    # sphere's own solref ("0.05 2") is consumed via geom mixing. An
    # explicit pair without solref attributes would fall back to o_solref
    # and never touch geom values, so this variant proves actual mixed
    # source constants by asserting the lowered hi differs from o_solref.
    xml_nomix = """
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
        <body name="probe" pos="0.45 0 0">
          <freejoint/><geom name="probe_sphere" type="sphere" size="0.1"
            solref="0.05 2"/></body>
      </worldbody>
    </mujoco>"""
    from mujoco_metal.coupled_constraints import lower_coupled_constraints
    _m = mujoco.MjModel.from_xml_string(xml_nomix)
    _desc = lower_coupled_constraints(_m)
    assert int(_desc.npairs) >= 1
    _o = np.asarray(_m.opt.o_solref, dtype=np.float32)
    assert not np.allclose(np.asarray(_desc.solref).reshape(-1, 2)[0], _o), (
        np.asarray(_desc.solref).reshape(-1, 2)[0], _o)
    # Stiffer mixed solref conditions the solve; force tier documents the
    # block-solver tolerance scale (E pairing itself holds the tight tier).
    _check_variant(xml_nomix, etier=5e-9, ftier=5e-6)
  elif variant == "mixed-analytic-sdf":
    # Add a touching analytic cube body instead: rebuild below.
    xml2 = """
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
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="probe" pos="0.45 0 0">
          <freejoint/><geom name="probe_sphere" type="sphere" size="0.1"/>
        </body>
        <body name="cube" pos="0 0 0.05">
          <freejoint/><geom name="box" type="box" size="0.05 0.05 0.05"/>
        </body>
      </worldbody>
      <contact><pair geom1="plugin_sdf" geom2="probe_sphere" condim="1"/>
      <pair geom1="floor" geom2="box"/></contact>
    </mujoco>"""
    _check_variant(xml2, etier=5e-9)
  elif variant == "moving-state":
    # float32 qvel input rounding amplified by B (~1e-6 per unit qvel)
    # dominates; still 100x below a missing-vel-term error (~50).
    _check_variant(_torus_xml(),
                   qvel=np.array([0.5, -0.25, 0.1, 0.05, -0.05, 0.02]),
                   etier=1e-5)
  elif variant == "batch-2":
    _check_variant(_torus_xml(), batch=2, etier=5e-9)
  elif variant == "contact-transition":
    # Batch with contact in world 0 and no contact in world 1 (far pose):
    # world 1 publishes empty mask with success; world 0 matches oracle.
    _check_contact_transition(_torus_xml(), etier=5e-9)
  elif variant == "clamp-equal-d0d1":
    # Distinct out-of-range raw d0/d1 clamp to the SAME value (0.9999):
    # the flat (d0==d1) branch must be taken. Pre-canonicalization the raw
    # predicates (10.0 != 20.0) took the lerp branch with unclamped
    # impedances; the paired E must match the clamped CPU oracle tightly.
    m = mujoco.MjModel.from_xml_string(_torus_xml())
    m.opt.enableflags |= flag
    m.opt.o_solimp[:] = (10.0, 20.0, 0.001, 0.5, 2.0)
    _check_variant_xml(m, 5e-9)
  elif variant == "clamp-low-bound":
    # All-below-range raw values clamp to lower bounds (d0/d1/mid to
    # 1e-4, width to 0, power to 1): flat branch with zero width. K ~ 1e8
    # at the 1e-4 impedance bound amplifies float32 input rounding, so
    # the tier documents the K-scaled bound (still far below branch-error
    # scale, which measured ~2e3 pre-canonicalization on the equal-clamp).
    m = mujoco.MjModel.from_xml_string(_torus_xml())
    m.opt.enableflags |= flag
    m.opt.o_solimp[:] = (-5.0, -2.0, -0.001, -1.0, 0.5)
    _check_variant_xml(m, 1e-8)
  else:
    raise AssertionError(variant)


def _check_variant_xml(model, etier, ftier=1e-4):
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  # Same represented state: CPU runs at float32(qpos) as float64.
  qpos32 = qpos.astype(np.float32).astype(np.float64)
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nslots = int(cc.descriptor.ncontacts_max)
  rd = cc._workspace["contact_row_data"].cpu().numpy().reshape(nslots,60)[:, :42].reshape(nslots,6,7)
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  d = mujoco.MjData(model)
  d.qpos[:] = qpos32
  d.qvel[:] = np.zeros(model.nv)
  mujoco.mj_forward(model, d)
  _aref = np.asarray(d.efc_aref, dtype=np.float64)
  cpu = [(float(d.contact[c].dist), np.asarray(d.contact[c].pos),
          float(_aref[int(d.contact[c].efc_address)]),
          int(d.contact[c].efc_address)) for c in range(d.ncon)]
  assert d.ncon > 0
  worst = 0.0
  used = set()
  for i in [x for x in range(nslots) if mask[x] > 0.5]:
    c = min((x for x in range(d.ncon) if x not in used),
            key=lambda x: float(np.linalg.norm(cpu[x][1] - npos[i])))
    used.add(c)
    assert float(np.linalg.norm(cpu[c][1] - npos[i])) < 1e-4
    worst = max(worst, abs(
        float(rd[i, 0, 3]) + float(rd[i, 0, 6]) - cpu[c][2]))
  assert worst < etier, worst
  # Publication consumes the same canonical constants: mapped force and
  # acceleration agree with the pinned oracle (no sorting; DOF-indexed).
  nat = np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64)
  cpu_f = np.asarray(d.qfrc_constraint, dtype=np.float64)
  scale = max(1.0, float(np.max(np.abs(cpu_f))))
  assert float(np.max(np.abs(nat - cpu_f))) / scale < ftier
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)


def _check_shallow(model, etier):
  # Bisect the probe outward until the CPU depth is just inside the
  # unsaturated impedance window (|d| < width), exercising lerp/pow.
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  d = mujoco.MjData(model)
  lo, hi = 0.45, 0.60
  for _ in range(24):
    mid = 0.5 * (lo + hi)
    qp = qpos.copy()
    qp[0] = mid
    d.qpos[:] = qp
    d.qvel[:] = np.zeros(model.nv)
    mujoco.mj_forward(model, d)
    if d.ncon > 0 and min(float(d.contact[c].dist)
                          for c in range(d.ncon)) < -1e-5:
      lo = mid
    else:
      hi = mid
  qp = qpos.copy()
  qp[0] = lo
  # Same represented state: the device consumes float32(qp); the oracle
  # runs at that rounded pose (not the unrounded bisection pose), so
  # geometry-input quantization is separated from formula tolerances.
  qp32 = qp.astype(np.float32).astype(np.float64)
  sim.reset(qpos=qp.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nslots = int(cc.descriptor.ncontacts_max)
  rd = cc._workspace["contact_row_data"].cpu().numpy().reshape(nslots,60)[:, :42].reshape(nslots,6,7)
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  d.qpos[:] = qp32
  d.qvel[:] = np.zeros(model.nv)
  mujoco.mj_forward(model, d)
  _aref = np.asarray(d.efc_aref, dtype=np.float64)
  cpu = [(float(d.contact[c].dist), np.asarray(d.contact[c].pos),
          float(_aref[int(d.contact[c].efc_address)]),
          int(d.contact[c].efc_address)) for c in range(d.ncon)]
  assert d.ncon > 0
  worst = 0.0
  used = set()
  for i in [x for x in range(nslots) if mask[x] > 0.5]:
    c = min((x for x in range(d.ncon) if x not in used),
            key=lambda x: float(np.linalg.norm(cpu[x][1] - npos[i])))
    used.add(c)
    assert float(np.linalg.norm(cpu[c][1] - npos[i])) < 1e-4
    worst = max(worst, abs(
        float(rd[i, 0, 3]) + float(rd[i, 0, 6]) - cpu[c][2]))
  assert worst < etier, worst


@_needs_gpu()
@pytest.mark.parametrize("condim", ["3", "4", "6"], ids=["c3", "c4", "c6"])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_friction_rows_mapped_with_moving_state_gpu(condim):  # Position-matched ALL-ROW comparison with nonzero tangent/spin/rolling
  # velocity: the 019 row-mapped oracle (J/R/ar/rhs/W per contact row plus
  # qacc/qfrc) runs on pyramidal condim 3/4/6 SDF fixtures with moving
  # states, so friction-row velocities, references and forces are
  # genuinely exercised (static presses leave them at zero). Nonzero
  # friction-row velocities and multipliers are asserted so the test
  # cannot pass on normal-row-only coverage.
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import (
      _cpu_contacts, _assert_contact_system_matches_cpu)
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_torus_xml(condim=condim))
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  qvel = np.array([0.15, -0.4, 0.1, 0.0, 0.0, 1.5])
  qvel32 = qvel.astype(np.float32).astype(np.float64)
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=qvel.astype(np.float32).reshape(1, -1))
  cpu_data, cpu_J, contacts = _cpu_contacts(model, qpos, qvel32)
  assert len(contacts) == 8
  assert all(len(c["rows"]) > 1 for c in contacts)
  _assert_contact_system_matches_cpu(model, sim, cpu_data, cpu_J, contacts)
  s = sim.assembled_system(recompute=True)
  lam = np.asarray(s["lambda"][0].cpu().numpy(), dtype=np.float64)
  desc = sim._coupled_constraints.descriptor
  packed = np.asarray(desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  from test_bundled_sdf_rigid_oracle_019 import _unique_native_slots
  assignment = _unique_native_slots(contacts, mask, npos)
  fric_vel, fric_lam = 0.0, 0.0
  for ci, contact in enumerate(contacts):
    slot, _ = assignment[ci]
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    rows = np.arange(gpu_base, gpu_base + len(contact["rows"]))
    fric_vel = max(fric_vel, float(np.max(np.abs(
        np.asarray(cpu_data.efc_vel, dtype=np.float64)[contact["rows"][1:]]))))
    fric_lam = max(fric_lam, float(np.max(np.abs(lam[rows[1:]]))))
  assert fric_vel > 1e-6, fric_vel
  assert fric_lam > 0.0, fric_lam


@_needs_gpu()
@pytest.mark.parametrize("which", ["solref", "solimp"],
                         ids=["iref", "iimp"])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_override_lifecycle_rebuild_gpu(which):
  # Documented lifecycle contract (no auto-recompile): mutating override
  # options after construction makes the next state/cache-touching call
  # fail BEFORE any state or workspace write (proving the binding, not a
  # stale silent solve); the documented `apply_lifecycle` rebuild then
  # consumes the mutated values and matches the oracle built from the
  # same mutated model.
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  flag = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
  m = mujoco.MjModel.from_xml_string(_torus_xml())
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  s0 = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  qpos_before = np.asarray(
      sim.state._qpos[0].cpu().numpy()).copy()
  # Post-construction mutation WITHOUT rebuild: the binding rejects the
  # next solve before touching state (no silent stale solve).
  m.opt.enableflags |= flag
  if which == "solref":
    m.opt.o_solref[:] = (0.05, 2.0)
  else:
    m.opt.o_solimp[:] = (0.85, 0.9, 0.002, 0.4, 3.0)
  with pytest.raises(ValueError, match="apply_lifecycle"):
    sim.assembled_system(recompute=True)
  # Fail-before-write covers positions, velocities, status and retained
  # warmstart (the state touched by a solve entry point); it does not
  # cover sensor samples or delay rings (untouched by assembly). Listed
  # buffers are asserted unchanged below, not "all state" in general.
  np.testing.assert_array_equal(
      np.asarray(sim.state._qpos[0].cpu().numpy()), qpos_before)
  np.testing.assert_array_equal(
      np.asarray(sim.state._qvel[0].cpu().numpy()),
      np.zeros((m.nv,), dtype=np.float32))
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  assert float(np.linalg.norm(sim.get_warmstart())) == 0.0
  # Documented rebuild consumes the mutated values.
  sim.apply_lifecycle(m)
  desc = sim._coupled_constraints.descriptor
  assert int(desc.npairs) >= 1
  if which == "solref":
    np.testing.assert_array_equal(
        np.asarray(desc.solref).reshape(-1, 2)[0],
        np.asarray(m.opt.o_solref, dtype=np.float32))
  else:
    np.testing.assert_array_equal(
        np.asarray(desc.solimp).reshape(-1, 5)[0],
        np.asarray(m.opt.o_solimp, dtype=np.float32))
  # A second solve proceeds without rebinding, and publication matches
  # the oracle built from the same mutated model.
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  d = mujoco.MjData(m)
  d.qpos[:] = qpos.astype(np.float32).astype(np.float64)
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  assert d.ncon > 0
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(
      np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qfrc_constraint, dtype=np.float64), rtol=1e-3, atol=1e-3)


@_needs_gpu()
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_elliptic_friction_rows_mapped_with_moving_state_gpu():
  # Distinct non-default elliptic solref/solreffriction with nonzero
  # tangent/spin velocity: the elliptic cone block (normal + tangent rows
  # with friction_B from solreffriction) is row-mapped against the oracle,
  # with nonzero friction-row velocities and multipliers asserted.
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import (
      _cpu_contacts, _assert_contact_system_matches_cpu,
      _unique_native_slots)
  from mujoco_metal.simulation import MetalSimulation
  flag = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
  m = mujoco.MjModel.from_xml_string(_torus_xml(condim="3",
                                               opt='cone="elliptic"'))
  m.opt.enableflags |= flag
  m.opt.o_solref[:] = (0.03, 1.5)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  qvel = np.array([0.1, -0.35, 0.05, 0.0, 1.2, -0.4])
  qvel32 = qvel.astype(np.float32).astype(np.float64)
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=qvel.astype(np.float32).reshape(1, -1))
  cpu_data, cpu_J, contacts = _cpu_contacts(m, qpos, qvel32)
  assert len(contacts) == 8
  assert all(len(c["rows"]) > 1 for c in contacts)
  _assert_contact_system_matches_cpu(m, sim, cpu_data, cpu_J, contacts)
  s = sim.assembled_system(recompute=True)
  lam = np.asarray(s["lambda"][0].cpu().numpy(), dtype=np.float64)
  desc = sim._coupled_constraints.descriptor
  packed = np.asarray(desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  assignment = _unique_native_slots(contacts, mask, npos)
  fric_vel, fric_lam = 0.0, 0.0
  for ci, contact in enumerate(contacts):
    slot, _ = assignment[ci]
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    rows = np.arange(gpu_base, gpu_base + len(contact["rows"]))
    fric_vel = max(fric_vel, float(np.max(np.abs(
        np.asarray(cpu_data.efc_vel, dtype=np.float64)[contact["rows"][1:]]))))
    fric_lam = max(fric_lam, float(np.max(np.abs(lam[rows[1:]]))))
  assert fric_vel > 1e-6, fric_vel
  assert fric_lam > 0.0, fric_lam


@_needs_gpu()
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_elliptic_distinct_reference_coefficients_mapped_gpu():
  # Distinct independently chosen normal/friction reference coefficients
  # via explicit pair attributes (solref vs solreffriction differ, not
  # override-coupled): elliptic cone with nonzero tangent/spin velocity
  # exercises friction_B from solreffriction while K/B come from solref.
  # Row-mapped against the oracle with nonzero friction asserts.
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import (
      _cpu_contacts, _assert_contact_system_matches_cpu,
      _unique_native_slots)
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  m = mujoco.MjModel.from_xml_string(_torus_xml(
      condim="3", opt='cone="elliptic"',
      pair_extra='solref="0.03 1.5" solreffriction="0.04 1.2"'))
  desc = lower_coupled_constraints(m)
  assert int(desc.npairs) >= 1
  np.testing.assert_array_equal(
      np.asarray(desc.solref).reshape(-1, 2)[0],
      np.array([0.03, 1.5], dtype=np.float32))
  np.testing.assert_array_equal(
      np.asarray(desc.solreffriction).reshape(-1, 2)[0],
      np.array([0.04, 1.2], dtype=np.float32))
  assert not np.allclose(np.asarray(desc.solref).reshape(-1, 2)[0],
                         np.asarray(desc.solreffriction).reshape(-1, 2)[0])
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(
      sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  qvel = np.array([0.2, 0.3, -0.1, 0.5, 0.0, -0.7])
  qvel32 = qvel.astype(np.float32).astype(np.float64)
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=qvel.astype(np.float32).reshape(1, -1))
  cpu_data, cpu_J, contacts = _cpu_contacts(m, qpos, qvel32)
  assert len(contacts) == 8
  assert all(len(c["rows"]) > 1 for c in contacts)
  _assert_contact_system_matches_cpu(m, sim, cpu_data, cpu_J, contacts)
  s = sim.assembled_system(recompute=True)
  lam = np.asarray(s["lambda"][0].cpu().numpy(), dtype=np.float64)
  packed = np.asarray(desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  assignment = _unique_native_slots(contacts, mask, npos)
  fric_vel, fric_lam = 0.0, 0.0
  for ci, contact in enumerate(contacts):
    slot, _ = assignment[ci]
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    rows = np.arange(gpu_base, gpu_base + len(contact["rows"]))
    fric_vel = max(fric_vel, float(np.max(np.abs(
        np.asarray(cpu_data.efc_vel, dtype=np.float64)[contact["rows"][1:]]))))
    fric_lam = max(fric_lam, float(np.max(np.abs(lam[rows[1:]]))))
  assert fric_vel > 1e-6, fric_vel
  assert fric_lam > 0.0, fric_lam
