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
  sim.reset(qpos=np.tile(qpos.astype(np.float32), (batch, 1)),
            qvel=np.tile(qvel.astype(np.float32), (batch, 1)))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nslots = int(cc.descriptor.ncontacts_max)
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  mujoco.mj_forward(model, d)
  _aref = np.asarray(d.efc_aref, dtype=np.float64)
  cpu = [(float(d.contact[c].dist), np.asarray(d.contact[c].pos),
          float(_aref[int(d.contact[c].efc_address)]),
          int(d.contact[c].efc_address)) for c in range(d.ncon)]
  assert len(cpu) == d.ncon and d.ncon > 0
  rdb = cc._workspace["contact_row_data"].cpu().numpy().reshape(
      batch, nslots, 6, 7)
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
  nat = np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64)
  cpu_f = np.asarray(d.qfrc_constraint, dtype=np.float64)
  scale = max(1.0, float(np.max(np.abs(cpu_f))))
  assert float(np.max(np.abs(nat - cpu_f))) / scale < ftier
  return worst


@_needs_gpu()
@pytest.mark.parametrize("variant", [
    "direct-solref",
    "refsafe-clamp",
    "power-1-lerp",
    "power-2-pow",
    "margin-gap",
    "condim-3-pyramidal",
    "condim-3-elliptic",
    "geom-mixing",
    "mixed-analytic-sdf",
    "moving-state",
    "batch-2",
    "clamp-equal-d0d1",
    "clamp-low-bound",
])
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
    m = mujoco.MjModel.from_xml_string(_torus_xml())
    m.opt.enableflags |= flag
    m.opt.o_margin = 0.002
    _check_variant_xml(m, 5e-9)
  elif variant == "condim-3-pyramidal":
    _check_variant(_torus_xml(condim="3"), etier=5e-9)
  elif variant == "condim-3-elliptic":
    # Cone-block solver tolerance scale for force (normal-row E stays
    # exact, proving the pairing; the force residual is solver tolerance).
    _check_variant(_torus_xml(condim="3", opt='cone="elliptic"'),
                   etier=5e-9, ftier=5e-6)
  elif variant == "geom-mixing":
    _check_variant(_torus_xml(sphere='<geom name="probe_sphere" '
                              'type="sphere" size="0.1" solref="0.05 2"/>'),
                   etier=5e-9)
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
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nslots = int(cc.descriptor.ncontacts_max)
  rd = cc._workspace["contact_row_data"].cpu().numpy().reshape(nslots, 6, 7)
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
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
  sim.reset(qpos=qp.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(
      sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nslots = int(cc.descriptor.ncontacts_max)
  rd = cc._workspace["contact_row_data"].cpu().numpy().reshape(nslots, 6, 7)
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  d.qpos[:] = qp
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
