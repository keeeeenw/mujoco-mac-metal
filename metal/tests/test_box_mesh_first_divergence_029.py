# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Box mesh-SDF first-divergence diagnostic at the exact 013 press pose.

Bounded reproducer plus structured same-input stage packets for
`box@(0.0,0.0)/press`. Methodology (same represented float32 gate pose on
both sides, one producer epoch, position-matched slots) is asserted; the
divergence packet is REPORTED with measured magnitudes:

- seeds agree closely (seed-1 init y differs ~2.6e-17 absolute);
- COLLISION terminals agree (x to 7 digits, dist to 8);
- the INTERSECTION gradient side is decided by a strict `< 0` comparison
  on a ~1e-18 residual (analytic-frame x of a point on the symmetry
  plane), which rounds oppositely on the two sides;
- the published press contact is an EXACT x-mirror (positions and normals
  mirrored in x, equal otherwise, dists equal).

REPAIR TARGET: the mirror assertions below characterize the current
divergence. A correct seed-tie-channel repair (bit-identical f64-ordered
seed evaluation) must replace them with direct native==CPU equality and
close the 013 pair-matrix gate; do not weaken them into pass-by-tolerance.
"""

import os

import numpy as np
import pytest


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


_GATE_Z = 0.06099992245435715
_GATE_QP = np.array([0.0, 0.0, _GATE_Z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _gate_model():
  import sys
  sys.path.insert(0, "metal/tests")
  from test_sdf_contact_013 import _model, _touch
  m = _model("box")
  zt = _touch(m, 0.0, 0.0)
  z = zt - 0.004
  # Touch height is float64-stable; the gate qp is its float32 cast.
  assert abs(float(zt) - 0.0649999237060543) < 1e-12, zt
  assert _GATE_QP[2] == np.float32(float(z)), (_GATE_QP[2], z)
  return m


@_needs_gpu()
def test_box_press_first_divergence_packet_gpu():
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from test_sdf_seed_source_trace_013 import (
      _capture_staged_workspace, _collapse_words, _collapse_wide,
      _STAGE, _STATE_WORDS)
  from mujoco_metal.simulation import MetalSimulation
  m = _gate_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=_GATE_QP.reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  epochs, _ = _capture_staged_workspace(sim)
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  # Methodology: exactly one complete producer epoch at this pose.
  assert len(epochs) == 1
  ep = epochs[-1]
  assert ep[0][0] == "_contact_sdf_seed_init"
  assert ep[-1][0] == "_contact_sdf_finalize_producer"
  # Same-input oracle at the identical represented pose.
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(_GATE_QP, dtype=np.float64)
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  assert int(d.ncon) == 2
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  ndist = np.asarray(s["contact_distance"][0].cpu().numpy())
  nnrm = np.asarray(s["contact_normal"][0].cpu().numpy())
  slots = [int(x) for x in range(len(mask)) if mask[x] > 0.5]
  assert len(slots) == 2
  # Position-matched slots; press contact is the deeper one on both sides.
  used = set()
  assignment = {}
  for i in slots:
    c = min((x for x in range(d.ncon) if x not in used),
            key=lambda x: float(np.linalg.norm(
                np.asarray(d.contact[x].pos) - npos[i])))
    used.add(c)
    assert float(np.linalg.norm(np.asarray(d.contact[c].pos) - npos[i])) < 1e-3
    assignment[c] = i
  cpu_press = min(range(d.ncon),
                  key=lambda c: float(d.contact[c].dist))
  assert float(d.contact[cpu_press].dist) < -1e-3
  nat_press = assignment[cpu_press]
  # Depths agree tightly; this is not a depth/area tie.
  assert abs(float(ndist[nat_press]) - float(d.contact[cpu_press].dist)) < 1e-9
  cpu_nrm = np.asarray(d.contact[cpu_press].frame[:3], dtype=np.float64)
  cpu_pos = np.asarray(d.contact[cpu_press].pos, dtype=np.float64)
  nat_nrm = np.asarray(nnrm[nat_press], dtype=np.float64)
  nat_pos = np.asarray(npos[nat_press], dtype=np.float64)
  # CPU ground truth at the gate pose (robust over +-1e-3 x shifts).
  np.testing.assert_allclose(
      cpu_nrm, np.array([0.12222085138013072, -0.18293366315697751,
                         -0.97549850762155732]),
      rtol=0, atol=1e-9)
  # MIRROR CHARACTERIZATION (repair target, see module docstring): native
  # terminal is the exact x-mirror of the CPU terminal.
  assert abs(nat_pos[0] + cpu_pos[0]) < 1e-9, (nat_pos, cpu_pos)
  # Non-mirrored components agree to float32 publication floor (~1e-8 on
  # these scales); far below the 12e-3 gate tier.
  np.testing.assert_allclose(nat_pos[1:], cpu_pos[1:], rtol=0, atol=2e-8)
  assert abs(nat_nrm[0] + cpu_nrm[0]) < 1e-9, (nat_nrm, cpu_nrm)
  np.testing.assert_allclose(nat_nrm[1:], cpu_nrm[1:], rtol=0, atol=2e-8)
  # Staged seed-1 INTERSECTION terminal: accepted alpha and dist agree
  # (same basin depth); only the transverse sign differs.
  terms = []
  for name, _, state, _, _, _ in ep:
    if "line_search" not in name:
      continue
    st = state.cpu().numpy().reshape(-1, _STATE_WORDS)
    x = _collapse_words(st[1, _STAGE["x"]:][:9])
    terms.append((float(_collapse_wide(st[1, _STAGE["alpha"]:][:3])),
                  float(_collapse_wide(st[1, _STAGE["dist"]:][:3])),
                  [float(v) for v in x]))
  assert terms
  print(f"BOXDIVERGE cpu_press_dist={float(d.contact[cpu_press].dist):.12f} "
        f"cpu_nrm={cpu_nrm.tolist()} nat_nrm={nat_nrm.tolist()} "
        f"cpu_pos={cpu_pos.tolist()} nat_pos={nat_pos.tolist()} "
        f"seed1_terms={terms[-3:]}", flush=True)
