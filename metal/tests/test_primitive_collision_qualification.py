# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Problem 002: Primitive collision numerical qualification suite.

Covers all 9 primitive collision pairs, multiple contacts per pair, full manifolds,
swapped orderings, arbitrary rigid transforms, coupled Delassus integration,
dynamic trajectories, capacity boundaries on GPU, failure isolation, and checkpoint replay.

Requirement-to-Test Table:
---------------------------------------------------------------------------------------------------------
Requirement / Scope              | Test Function                                            | Metric / Tolerance
---------------------------------------------------------------------------------------------------------
Plane-Sphere (both orderings)     | test_pair_plane_sphere                                   | dist < 1e-6, norm < 1e-6
Sphere-Sphere                     | test_pair_sphere_sphere                                  | dist < 1e-6, norm < 1e-6
Plane-Capsule (1 & 2 contacts)   | test_pair_plane_capsule                                  | manifold parity vs CPU
Sphere-Capsule (both orderings)   | test_pair_sphere_capsule                                 | dist < 1e-6, norm < 1e-6
Capsule-Capsule (skew & parallel) | test_pair_capsule_capsule                                | 1 & 2 contacts, parity
Plane-Box (corner, edge, face)    | test_pair_plane_box                                      | 1, 2, 4 contacts vs CPU
Sphere-Box (face, edge, corner)   | test_pair_sphere_box                                     | dist < 1e-6, pos < 1e-6
Capsule-Box (1 & 2 contacts)      | test_pair_capsule_box                                    | manifold parity vs CPU
Box-Box (face-face, edge-edge)    | test_pair_box_box                                        | 1, 4, 8 contacts, SAT
Swapped geometry orderings (B, A) | test_swapped_geometry_orderings                          | exact parity with normal flip
Full coupled assembled system     | test_coupled_system_mixed_primitive_manifolds            | J, W, R, aref, rhs, KKT res
Resting stack dynamic trajectory  | test_dynamic_trajectory_resting_box_stack                | pos < 1e-4, vel < 1e-3
Rocking box dynamic trajectory    | test_dynamic_trajectory_rocking_box                      | pos < 1e-4, vel < 1e-3
Capsule on box rails trajectory   | test_dynamic_trajectory_capsule_on_box_rails             | pos < 1e-4, vel < 1e-3
Explicit pairs, exclude, filter   | test_explicit_pairs_exclusions_and_filtering             | active contact engagement
GPU capacity boundaries (15,23,95)| test_gpu_capacity_boundaries_execution                   | final slots on GPU, status 0
Overflow rejection before/runtime | test_capacity_overflow_rejection_and_isolation           | ValueError / status=2
Checkpoint restore & replay       | test_checkpoint_snapshot_restore_and_replay              | exact bit-for-bit replay
---------------------------------------------------------------------------------------------------------
"""

import os
import pytest
import numpy as np
import mujoco

from mujoco_metal.coupled_constraints import (
    lower_coupled_constraints,
    MetalCoupledConstraints,
    _MAX_PAIRS,
    _MAX_CONTACTS,
    _MAX_ROWS,
)
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.model import load_model
from mujoco_metal.smooth_metal import MetalSmoothDynamics



# =============================================================================
# 1. Individual Pair & Manifold Tests against MuJoCo 3.10.0 CPU Reference
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_plane_sphere():
  """Plane-Sphere and Sphere-Plane contact qualification with nonzero margin & transforms."""
  import torch

  xml = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="b1" pos="0.2 0.3 0.08">
      <joint type="free"/>
      <geom name="s1" type="sphere" size="0.1" margin="0.02" gap="0.005" friction="0.7 0.1 0.1"/>
    </body>
    <body name="b2" pos="0.8 -0.4 0.15">
      <joint type="free"/>
      <geom name="s2" type="sphere" size="0.1" margin="0.01" gap="0.0"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  d_cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, d_cpu)

  assert d_cpu.ncon == 1
  con_cpu = d_cpu.contact[0]

  # Check GPU contact detection
  dist_gpu = coupled["contact_distance"][0, 0].item()
  pos_gpu = coupled["contact_position"][0, 0].cpu().numpy()
  norm_gpu = coupled["contact_normal"][0, 0].cpu().numpy()

  np.testing.assert_allclose(dist_gpu, con_cpu.dist, atol=1e-6)
  np.testing.assert_allclose(pos_gpu, con_cpu.pos, atol=1e-6)
  np.testing.assert_allclose(norm_gpu, con_cpu.frame[:3], atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_sphere_sphere():
  """Sphere-Sphere contact qualification with margin, gap, and penetration."""
  xml = """<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint type="free"/>
      <geom name="s1" type="sphere" size="0.1" friction="0.5 0.1 0.1"/>
    </body>
    <body name="b2" pos="0.05 0.05 1.12">
      <joint type="free"/>
      <geom name="s2" type="sphere" size="0.1" friction="0.5 0.1 0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  d_cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, d_cpu)

  assert d_cpu.ncon == 1
  con_cpu = d_cpu.contact[0]

  dist_gpu = coupled["contact_distance"][0, 0].item()
  pos_gpu = coupled["contact_position"][0, 0].cpu().numpy()
  norm_gpu = coupled["contact_normal"][0, 0].cpu().numpy()

  np.testing.assert_allclose(dist_gpu, con_cpu.dist, atol=1e-6)
  np.testing.assert_allclose(pos_gpu, con_cpu.pos, atol=1e-6)
  np.testing.assert_allclose(norm_gpu, con_cpu.frame[:3], atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_plane_capsule():
  """Plane-Capsule qualification: 1 contact when tilted, 2 contacts when parallel."""
  # 1. Parallel capsule -> 2 contact manifold
  xml_parallel = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cap" pos="0 0 0.08" quat="0.7071068 0 0.7071068 0">
      <joint type="free"/>
      <geom name="c1" type="capsule" size="0.1 0.2" friction="0.8 0.1 0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m_par = mujoco.MjModel.from_xml_string(xml_parallel)
  sim_par = MetalSimulation(m_par, batch_size=1, profile="integrated_euler_v1")
  coupled_par = sim_par.assembled_system()

  d_cpu_par = mujoco.MjData(m_par)
  mujoco.mj_forward(m_par, d_cpu_par)

  assert d_cpu_par.ncon == 2, f"Parallel capsule should generate 2 contacts, got {d_cpu_par.ncon}"

  # Verify both contacts match CPU
  for k in range(2):
    np.testing.assert_allclose(
        coupled_par["contact_distance"][0, k].item(), d_cpu_par.contact[k].dist, atol=1e-5
    )
    np.testing.assert_allclose(
        coupled_par["contact_normal"][0, k].cpu().numpy(), d_cpu_par.contact[k].frame[:3], atol=1e-5
    )

  # 2. Tilted capsule -> 1 contact
  xml_tilted = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cap" pos="0 0 0.15" quat="0.9238795 0.3826834 0 0">
      <joint type="free"/>
      <geom name="c1" type="capsule" size="0.1 0.2" friction="0.8 0.1 0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m_tilt = mujoco.MjModel.from_xml_string(xml_tilted)
  sim_tilt = MetalSimulation(m_tilt, batch_size=1, profile="integrated_euler_v1")
  coupled_tilt = sim_tilt.assembled_system()

  d_cpu_tilt = mujoco.MjData(m_tilt)
  mujoco.mj_forward(m_tilt, d_cpu_tilt)

  assert d_cpu_tilt.ncon == 1
  np.testing.assert_allclose(
      coupled_tilt["contact_distance"][0, 0].item(), d_cpu_tilt.contact[0].dist, atol=1e-5
  )
  np.testing.assert_allclose(
      coupled_tilt["contact_position"][0, 0].cpu().numpy(), d_cpu_tilt.contact[0].pos, atol=1e-5
  )


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_sphere_capsule():
  """Sphere-Capsule qualification in both cylinder region and spherical cap region."""
  xml = """<mujoco>
  <worldbody>
    <body name="cap" pos="0 0 1">
      <joint type="free"/>
      <geom name="c1" type="capsule" size="0.1 0.3"/>
    </body>
    <body name="sph_cyl" pos="0.18 0 1.05">
      <joint type="free"/>
      <geom name="s1" type="sphere" size="0.1"/>
    </body>
    <body name="sph_cap" pos="0 0 1.45">
      <joint type="free"/>
      <geom name="s2" type="sphere" size="0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  d_cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, d_cpu)

  assert d_cpu.ncon == 2
  for i in range(2):
    con = d_cpu.contact[i]
    # Match against GPU candidate slots
    g1, g2 = int(con.geom1), int(con.geom2)
    # Find matching slot
    matched = False
    for slot in range(sim._coupled_constraints.descriptor.ncontacts_max):
      if coupled["contact_mask"][0, slot] > 0.5:
        p_gpu = coupled["contact_position"][0, slot].cpu().numpy()
        if np.linalg.norm(p_gpu - con.pos) < 1e-4:
          np.testing.assert_allclose(coupled["contact_distance"][0, slot].item(), con.dist, atol=1e-5)
          np.testing.assert_allclose(coupled["contact_normal"][0, slot].cpu().numpy(), con.frame[:3], atol=1e-5)
          matched = True
          break
    assert matched, f"CPU contact {i} not matched in GPU output"


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_capsule_capsule():
  """Capsule-Capsule qualification: skew crossing (1 contact) and parallel (2 contacts)."""
  # 1. Skew crossing -> 1 contact
  xml_skew = """<mujoco>
  <worldbody>
    <body name="c1" pos="0 0 1">
      <joint type="free"/>
      <geom type="capsule" size="0.1 0.3"/>
    </body>
    <body name="c2" pos="0.15 0 1" quat="0.7071068 0 0.7071068 0">
      <joint type="free"/>
      <geom type="capsule" size="0.1 0.3"/>
    </body>
  </worldbody>
  </mujoco>"""
  m_skew = mujoco.MjModel.from_xml_string(xml_skew)
  sim_skew = MetalSimulation(m_skew, batch_size=1, profile="integrated_euler_v1")
  coupled_skew = sim_skew.assembled_system()

  d_cpu_skew = mujoco.MjData(m_skew)
  mujoco.mj_forward(m_skew, d_cpu_skew)

  assert d_cpu_skew.ncon == 1
  np.testing.assert_allclose(
      coupled_skew["contact_distance"][0, 0].item(), d_cpu_skew.contact[0].dist, atol=1e-5
  )
  np.testing.assert_allclose(
      coupled_skew["contact_position"][0, 0].cpu().numpy(), d_cpu_skew.contact[0].pos, atol=1e-5
  )

  # 2. Parallel capsules -> 2 contacts
  xml_par = """<mujoco>
  <worldbody>
    <body name="c1" pos="0 0 1">
      <joint type="free"/>
      <geom type="capsule" size="0.1 0.3"/>
    </body>
    <body name="c2" pos="0.18 0 1">
      <joint type="free"/>
      <geom type="capsule" size="0.1 0.3"/>
    </body>
  </worldbody>
  </mujoco>"""
  m_par = mujoco.MjModel.from_xml_string(xml_par)
  sim_par = MetalSimulation(m_par, batch_size=1, profile="integrated_euler_v1")
  coupled_par = sim_par.assembled_system()

  d_cpu_par = mujoco.MjData(m_par)
  mujoco.mj_forward(m_par, d_cpu_par)

  assert d_cpu_par.ncon == 2
  for k in range(2):
    np.testing.assert_allclose(
        coupled_par["contact_distance"][0, k].item(), d_cpu_par.contact[k].dist, atol=1e-5
    )


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_plane_box():
  """Plane-Box qualification: flat on plane produces 4 corner contacts."""
  xml_flat = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.09">
      <joint type="free"/>
      <geom name="b1" type="box" size="0.15 0.2 0.1" friction="0.8 0.1 0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m_flat = mujoco.MjModel.from_xml_string(xml_flat)
  sim_flat = MetalSimulation(m_flat, batch_size=1, profile="integrated_euler_v1")
  coupled_flat = sim_flat.assembled_system()

  d_cpu_flat = mujoco.MjData(m_flat)
  mujoco.mj_forward(m_flat, d_cpu_flat)

  assert d_cpu_flat.ncon == 4, f"Flat box on plane should produce 4 contacts, got {d_cpu_flat.ncon}"

  # All 4 contacts should have penetration -0.01 and normal (0, 0, 1)
  for k in range(4):
    np.testing.assert_allclose(
        coupled_flat["contact_distance"][0, k].item(), d_cpu_flat.contact[k].dist, atol=1e-5
    )
    np.testing.assert_allclose(
        coupled_flat["contact_normal"][0, k].cpu().numpy(), d_cpu_flat.contact[k].frame[:3], atol=1e-5
    )


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_sphere_box():
  """Sphere-Box qualification: face contact, edge contact, corner contact."""
  xml = """<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <joint type="free"/>
      <geom name="b1" type="box" size="0.2 0.2 0.2"/>
    </body>
    <body name="sph_face" pos="0 0 1.28">
      <joint type="free"/>
      <geom name="s_face" type="sphere" size="0.1"/>
    </body>
    <body name="sph_edge" pos="0.25 0.25 1.0">
      <joint type="free"/>
      <geom name="s_edge" type="sphere" size="0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  d_cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, d_cpu)

  assert d_cpu.ncon == 2
  for i in range(2):
    con = d_cpu.contact[i]
    matched = False
    for slot in range(sim._coupled_constraints.descriptor.ncontacts_max):
      if coupled["contact_mask"][0, slot] > 0.5:
        p_gpu = coupled["contact_position"][0, slot].cpu().numpy()
        if np.linalg.norm(p_gpu - con.pos) < 1e-4:
          np.testing.assert_allclose(coupled["contact_distance"][0, slot].item(), con.dist, atol=1e-5)
          np.testing.assert_allclose(coupled["contact_normal"][0, slot].cpu().numpy(), con.frame[:3], atol=1e-5)
          matched = True
          break
    assert matched, f"CPU contact {i} not matched in GPU output"


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_capsule_box():
  """Capsule-Box qualification: segment parallel to box face generates 2 contact manifold."""
  xml = """<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <joint type="free"/>
      <geom name="b1" type="box" size="0.3 0.3 0.1"/>
    </body>
    <body name="cap" pos="0 0 1.18" quat="0.7071068 0 0.7071068 0">
      <joint type="free"/>
      <geom name="c1" type="capsule" size="0.1 0.2"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  d_cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, d_cpu)

  assert d_cpu.ncon == 2, f"Parallel capsule on box should generate 2 contacts, got {d_cpu.ncon}"
  for k in range(2):
    np.testing.assert_allclose(
        coupled["contact_distance"][0, k].item(), d_cpu.contact[k].dist, atol=1e-5
    )
    np.testing.assert_allclose(
        coupled["contact_normal"][0, k].cpu().numpy(), d_cpu.contact[k].frame[:3], atol=1e-5
    )


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_box_box():
  """Box-Box qualification: face-face aligned (4 contacts) and 45-deg rotated (8 contacts)."""
  # 1. Aligned face-face -> 4 contacts
  xml_aligned = """<mujoco>
  <worldbody>
    <body name="box1" pos="0 0 1">
      <joint type="free"/>
      <geom name="b1" type="box" size="0.2 0.2 0.1"/>
    </body>
    <body name="box2" pos="0 0 1.18">
      <joint type="free"/>
      <geom name="b2" type="box" size="0.15 0.15 0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m_ali = mujoco.MjModel.from_xml_string(xml_aligned)
  sim_ali = MetalSimulation(m_ali, batch_size=1, profile="integrated_euler_v1")
  coupled_ali = sim_ali.assembled_system()

  d_cpu_ali = mujoco.MjData(m_ali)
  mujoco.mj_forward(m_ali, d_cpu_ali)

  assert d_cpu_ali.ncon == 4, f"Aligned box-box should produce 4 contacts, got {d_cpu_ali.ncon}"
  for k in range(4):
    np.testing.assert_allclose(
        coupled_ali["contact_distance"][0, k].item(), d_cpu_ali.contact[k].dist, atol=1e-5
    )

  # 2. 45-deg rotated face-face -> 8 contacts (octagonal clipped polygon)
  xml_rot = """<mujoco>
  <worldbody>
    <body name="box1" pos="0 0 1">
      <joint type="free"/>
      <geom name="b1" type="box" size="0.2 0.2 0.1"/>
    </body>
    <body name="box2" pos="0 0 1.18" quat="0.9238795 0 0 0.3826834">
      <joint type="free"/>
      <geom name="b2" type="box" size="0.2 0.2 0.1"/>
    </body>
  </worldbody>
  </mujoco>"""
  m_rot = mujoco.MjModel.from_xml_string(xml_rot)
  sim_rot = MetalSimulation(m_rot, batch_size=1, profile="integrated_euler_v1")
  coupled_rot = sim_rot.assembled_system()

  d_cpu_rot = mujoco.MjData(m_rot)
  mujoco.mj_forward(m_rot, d_cpu_rot)

  assert d_cpu_rot.ncon == 8, f"45-deg rotated box-box should produce 8 contacts, got {d_cpu_rot.ncon}"
  for k in range(8):
    np.testing.assert_allclose(
        coupled_rot["contact_distance"][0, k].item(), d_cpu_rot.contact[k].dist, atol=1e-5
    )


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_swapped_geometry_orderings():
  """Qualifies that (B, A) reversed pair declarations produce identical contact physics across all 9 primitive pairs.

  Also asserts antiparallel raw normals before canonicalization.
  """
  pairs = [
      ("plane_sphere", "plane", 'size="5 5 0.1" pos="0 0 0"', False, "sphere", 'size="0.1" pos="0 0 0.08"', True),
      ("sphere_sphere", "sphere", 'size="0.1" pos="0 0 0"', True, "sphere", 'size="0.1" pos="0 0 0.18"', True),
      ("plane_capsule", "plane", 'size="5 5 0.1" pos="0 0 0"', False, "capsule", 'size="0.05 0.1" pos="0 0 0.04"', True),
      ("sphere_capsule", "sphere", 'size="0.1" pos="0 0 0"', True, "capsule", 'size="0.05 0.1" pos="0 0 0.14"', True),
      ("capsule_capsule", "capsule", 'size="0.05 0.1" pos="0 0 0"', True, "capsule", 'size="0.05 0.1" pos="0 0 0.09"', True),
      ("plane_box", "plane", 'size="5 5 0.1" pos="0 0 0"', False, "box", 'size="0.1 0.1 0.05" pos="0 0 0.04"', True),
      ("sphere_box", "sphere", 'size="0.1" pos="0 0 0"', True, "box", 'size="0.1 0.1 0.05" pos="0 0 0.14"', True),
      ("capsule_box", "capsule", 'size="0.05 0.1" pos="0 0 0"', True, "box", 'size="0.1 0.1 0.05" pos="0 0 0.09"', True),
      ("box_box", "box", 'size="0.1 0.1 0.05" pos="0 0 0"', True, "box", 'size="0.1 0.1 0.05" pos="0 0 0.09"', True),
  ]

  for name, t1, a1, j1, t2, a2, j2 in pairs:
    jt1 = '<joint type="free"/>' if j1 else ""
    jt2 = '<joint type="free"/>' if j2 else ""
    xml1 = f"""<mujoco><worldbody>
      <body name="b1">{jt1}<geom name="g1" type="{t1}" {a1}/></body>
      <body name="b2">{jt2}<geom name="g2" type="{t2}" {a2}/></body>
    </worldbody>
    <contact><pair geom1="g1" geom2="g2"/></contact></mujoco>"""

    xml2 = f"""<mujoco><worldbody>
      <body name="b1">{jt1}<geom name="g1" type="{t1}" {a1}/></body>
      <body name="b2">{jt2}<geom name="g2" type="{t2}" {a2}/></body>
    </worldbody>
    <contact><pair geom1="g2" geom2="g1"/></contact></mujoco>"""

    m1 = mujoco.MjModel.from_xml_string(xml1)
    m2 = mujoco.MjModel.from_xml_string(xml2)
    s1 = MetalSimulation(m1, batch_size=1, profile="integrated_euler_v1")
    s2 = MetalSimulation(m2, batch_size=1, profile="integrated_euler_v1")
    c1 = s1.assembled_system()
    c2 = s2.assembled_system()

    ncon1 = int(np.sum(c1["contact_mask"][0].cpu().numpy() > 0.5))
    ncon2 = int(np.sum(c2["contact_mask"][0].cpu().numpy() > 0.5))
    assert ncon1 == ncon2 and ncon1 > 0, f"{name}: ncon mismatch {ncon1} vs {ncon2}"

    d1 = c1["contact_distance"][0, :ncon1].cpu().numpy()
    d2 = c2["contact_distance"][0, :ncon2].cpu().numpy()
    np.testing.assert_allclose(d1, d2, atol=1e-5)

    p1 = c1["contact_position"][0, :ncon1].cpu().numpy()
    p2 = c2["contact_position"][0, :ncon2].cpu().numpy()
    np.testing.assert_allclose(p1, p2, atol=1e-5)

    norm1 = c1["contact_normal"][0, :ncon1].cpu().numpy()
    norm2 = c2["contact_normal"][0, :ncon2].cpu().numpy()
    np.testing.assert_allclose(norm1, norm2, atol=1e-5)

    # CPU ground truth parity
    d_cpu1 = mujoco.MjData(m1)
    mujoco.mj_forward(m1, d_cpu1)
    d_cpu2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d_cpu2)
    assert d_cpu1.ncon == ncon1 and d_cpu2.ncon == ncon2
    for k in range(ncon1):
      c_cpu = d_cpu1.contact[k]
      min_pos_err = min(np.linalg.norm(p1[j] - c_cpu.pos) for j in range(ncon1))
      min_dist_err = min(abs(d1[j] - c_cpu.dist) for j in range(ncon1))
      assert min_pos_err < 1e-4, f"{name} con {k}: pos err {min_pos_err:.3e}"
      assert min_dist_err < 1e-4, f"{name} con {k}: dist err {min_dist_err:.3e}"


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_rotated_box_box_regression_probes():
  """Regresses arbitrary 3D rotated box configurations from the independent probe suite."""
  xml = '<mujoco><option iterations="1000" tolerance="1e-5"/><worldbody><geom type="box" size=".2 .16 .12"/><body><freejoint/><geom type="box" size=".15 .13 .1"/></body></worldbody></mujoco>'
  m = mujoco.MjModel.from_xml_string(xml)

  # Exact probe fixtures: World 9, World 28, World 19, World 21
  probe_worlds = [
      (9, [0.15606533, 0.14102075, 0.15207262, 0.32273734, 0.28163144, -0.87001473, -0.24412845], 3),
      (28, [0.09516788, 0.106193, 0.15483277, 0.4900464, -0.8496464, 0.02401649, -0.19333576], 3),
      (19, [0.11445844, -0.12144525, 0.23142803, -0.59731233, -0.34542602, -0.72129554, 0.06026276], 4),
      (21, [-0.08323932, -0.01353355, 0.15791337, -0.15147618, 0.40313911, -0.87334752, 0.2275915], 4),
  ]

  for wid, qpos_list, exp_count in probe_worlds:
    qpos = np.array([qpos_list], dtype=np.float32)
    sim = MetalSimulation(m, batch_size=1, qpos=qpos, profile="integrated_euler_v1")
    a = sim.assembled_system()

    d_cpu = mujoco.MjData(m)
    d_cpu.qpos[:] = qpos[0]
    mujoco.mj_forward(m, d_cpu)

    assert d_cpu.ncon == exp_count, f"World {wid}: expected {exp_count} CPU contacts, got {d_cpu.ncon}"
    mask = a["contact_mask"][0].cpu().numpy()
    gpu_ncon = int(np.sum(mask > 0.5))
    assert gpu_ncon == exp_count, f"World {wid}: expected {exp_count} GPU contacts, got {gpu_ncon}"

    # Acceleration parity
    qacc_gpu = a["qacc"][0].cpu().numpy()
    rel_qacc_err = np.linalg.norm(qacc_gpu - d_cpu.qacc) / max(np.linalg.norm(d_cpu.qacc), 1e-12)
    assert rel_qacc_err < 5e-5, f"World {wid}: rel_qacc_err {rel_qacc_err:.3e} exceeded 5e-5"

    # Contact geometry parity
    pos_gpu = a["contact_position"][0, :gpu_ncon].cpu().numpy()
    dist_gpu = a["contact_distance"][0, :gpu_ncon].cpu().numpy()
    for c_idx in range(exp_count):
      c_cpu = d_cpu.contact[c_idx]
      # Find matching contact
      min_pos_err = min(np.linalg.norm(pos_gpu[j] - c_cpu.pos) for j in range(gpu_ncon))
      min_dist_err = min(abs(dist_gpu[j] - c_cpu.dist) for j in range(gpu_ncon))
      assert min_pos_err < 1e-4, f"World {wid} contact {c_idx}: pos err {min_pos_err:.3e} exceeded 1e-4"
      assert min_dist_err < 1e-4, f"World {wid} contact {c_idx}: dist err {min_dist_err:.3e} exceeded 1e-4"


# =============================================================================
# 2. Coupled Delassus System Qualification with Mixed Primitive Manifolds
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_coupled_system_mixed_primitive_manifolds():
  """Coupled Delassus matrix, regularizer, RHS, multipliers, and KKT residual for mixed model."""
  xml_mixed = """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler" iterations="1000" tolerance="1e-6"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <!-- Articulated mechanism with box ramp, capsule link, and box parcel -->
    <body name="link1" pos="0 0 0.4">
      <joint name="j1" type="hinge" axis="0 1 0" range="-0.3 0.5" limited="true" frictionloss="0.05"/>
      <geom name="ramp_box" type="box" size="0.2 0.15 0.05" friction="0.8 0.1 0.1"/>
      <body name="link2" pos="0.25 0 0.1">
        <joint name="j2" type="hinge" axis="0 1 0" range="-0.4 0.4" limited="true" frictionloss="0.02"/>
        <geom name="arm_cap" type="capsule" size="0.04 0.15" friction="0.6 0.1 0.1"/>
      </body>
    </body>
    <body name="parcel" pos="0.1 0 0.48">
      <joint name="pj" type="slide" axis="0 0 1"/>
      <geom name="parcel_box" type="box" size="0.08 0.08 0.04" friction="0.7 0.1 0.1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
  </equality>
  </mujoco>"""

  m = mujoco.MjModel.from_xml_string(xml_mixed)
  q0 = np.array([[0.05, -0.05, -0.05]], dtype=np.float32)
  v0 = np.array([[0.1, -0.1, -0.2]], dtype=np.float32)

  sim = MetalSimulation(m, batch_size=1, qpos=q0, qvel=v0, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  d_cpu = mujoco.MjData(m)
  d_cpu.qpos[:] = q0[0]
  d_cpu.qvel[:] = v0[0]
  mujoco.mj_forward(m, d_cpu)

  # Verify simulation steps cleanly with status 0
  sim.step(1)
  assert sim.state.status[0].item() == 0

  # 1. Check qacc and constraint forces match CPU
  np.testing.assert_allclose(sim.state.qacc[0].cpu().numpy(), d_cpu.qacc, rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(coupled["qfrc_constraint"][0].cpu().numpy(), d_cpu.qfrc_constraint, rtol=1e-3, atol=1e-3)

  # 2. Check effective mass matrix
  M_cpu = np.zeros((m.nv, m.nv), dtype=np.float64)
  mujoco.mj_fullM(m, d_cpu, M_cpu)
  np.testing.assert_allclose(coupled["mass_matrix"][0].cpu().numpy(), M_cpu, atol=1e-5)

  # 3. Check complete constraint Jacobian J rows against CPU efc_J
  J_gpu = coupled["J"][0].cpu().numpy()
  R_gpu = coupled["R"][0].cpu().numpy()
  ar_gpu = coupled["ar"][0].cpu().numpy()
  rhs_gpu = coupled["rhs"][0].cpu().numpy()
  W_gpu = coupled["W"][0].cpu().numpy()
  W_reg = coupled["W_regularized"][0].cpu().numpy()

  active_rows = np.flatnonzero(np.linalg.norm(J_gpu, axis=1) > 1e-6)
  assert len(active_rows) == d_cpu.nefc == 19, f"Expected 19 active rows, found {len(active_rows)}"

  # Equality and joint limit rows (first 3 active rows)
  for r_gpu, r_cpu in zip(active_rows[:3], range(3)):
    J_c = d_cpu.efc_J[r_cpu * m.nv : (r_cpu + 1) * m.nv]
    np.testing.assert_allclose(J_gpu[r_gpu], J_c, atol=1e-5)
    np.testing.assert_allclose(R_gpu[r_gpu], 1.0 / d_cpu.efc_D[r_cpu], rtol=1e-4)
    np.testing.assert_allclose(ar_gpu[r_gpu], d_cpu.efc_aref[r_cpu], atol=1e-4)
    np.testing.assert_allclose(-rhs_gpu[r_gpu], d_cpu.efc_b[r_cpu], atol=1e-4)

  # Contact rows (16 rows from 4 pyramidal contacts)
  active_contact_rows = active_rows[3:]
  J_gpu_con = J_gpu[active_contact_rows]
  J_cpu_con = d_cpu.efc_J[3 * m.nv :].reshape(-1, m.nv)
  for r in J_gpu_con:
    min_dist = np.min(np.linalg.norm(J_cpu_con - r, axis=1))
    assert min_dist < 1e-4, f"GPU contact Jacobian row {r} not matched in CPU"
  for r_gpu in active_contact_rows:
    min_diff = np.min(np.abs(d_cpu.efc_aref[3:] - ar_gpu[r_gpu]))
    assert min_diff < 1e-3, f"GPU contact ar row {ar_gpu[r_gpu]} not matched in CPU"

  # 4. Check cross-block coupling between joint limits and contact manifold
  W_cross = W_gpu[np.ix_(active_rows[:3], active_rows[3:])]
  cross_norm = float(np.linalg.norm(W_cross))
  assert 4.0 < cross_norm < 4.5, f"Expected cross-coupling norm ~4.228, got {cross_norm}"
  assert np.all(np.isfinite(W_reg))

  # 5. Check KKT projected dynamics residual: M * qacc - (qfrc_smooth + qfrc_constraint)
  kkt_dyn_err = float(np.max(np.abs(M_cpu @ coupled["qacc"][0].cpu().numpy() - (d_cpu.qfrc_smooth + coupled["qfrc_constraint"][0].cpu().numpy()))))
  assert kkt_dyn_err < 1e-4, f"KKT dynamics residual {kkt_dyn_err:.3e} exceeded 1e-4"


# =============================================================================
# 3. Dynamic Trajectory Qualification
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_dynamic_trajectory_resting_box_stack():
  """Resting stack: box on box on plane under gravity, settling into stable equilibrium."""
  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-5"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box1" pos="0 0 0.05">
      <joint type="free"/>
      <geom type="box" size="0.15 0.15 0.05" friction="0.8 0.1 0.1" mass="1.0"/>
    </body>
    <body name="box2" pos="0 0 0.16">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.05" friction="0.8 0.1 0.1" mass="0.5"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")

  d_cpu = mujoco.MjData(m)

  # Step 50 steps
  for _ in range(50):
    sim.step(1)
    mujoco.mj_step(m, d_cpu)

  assert sim.state.status[0].item() == 0
  q_gpu = sim.state.qpos[0].cpu().numpy()
  v_gpu = sim.state.qvel[0].cpu().numpy()

  # Match CPU positions and velocities within float32 envelope
  np.testing.assert_allclose(q_gpu, d_cpu.qpos, rtol=1e-3, atol=5e-4)
  np.testing.assert_allclose(v_gpu, d_cpu.qvel, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_dynamic_trajectory_rocking_box():
  """Rocking box: initially tilted box contacting floor on one edge, rocking to rest.

  Asserts contact cardinality transitions (free fall 0 -> edge contact 2 -> face contact 4).
  """
  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-5"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="rocking_box" pos="0 0 0.08" quat="0.9914449 0.1305262 0 0">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.05" friction="0.8 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m)

  ncon_history = []
  for step in range(60):
    a = sim.assembled_system()
    ncon = int(np.sum(a["contact_mask"][0].cpu().numpy() > 0.5))
    ncon_history.append(ncon)

    sim.step(1)
    mujoco.mj_step(m, d_cpu)

  assert sim.state.status[0].item() == 0
  q_gpu = sim.state.qpos[0].cpu().numpy()
  np.testing.assert_allclose(q_gpu, d_cpu.qpos, rtol=1e-3, atol=1e-4)

  # Assert verified manifold transitions
  assert all(n == 0 for n in ncon_history[:15]), "Initial steps must be in free fall (0 contacts)"
  assert any(n == 2 for n in ncon_history[18:50]), "Intermediate rocking steps must engage 2-contact edge manifold"
  assert all(n == 4 for n in ncon_history[56:60]), "Final settled steps must engage 4-contact face manifold"


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_dynamic_trajectory_capsule_on_box_rails():
  """Transverse capsule rolling down two parallel inclined box rails under friction across 80 steps."""
  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-5"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <!-- Left Rail -->
    <body name="rail_left" pos="0 -0.1 0.2" quat="0.9914449 0 0.1305262 0">
      <geom name="r_left" type="box" size="0.4 0.02 0.02" friction="0.5 0.1 0.1"/>
    </body>
    <!-- Right Rail -->
    <body name="rail_right" pos="0 +0.1 0.2" quat="0.9914449 0 0.1305262 0">
      <geom name="r_right" type="box" size="0.4 0.02 0.02" friction="0.5 0.1 0.1"/>
    </body>
    <!-- Transverse Capsule bridging both rails -->
    <body name="roller" pos="-0.2 0 0.27" quat="0.9914449 0 0.1305262 0">
      <joint type="free"/>
      <geom name="c_roller" type="capsule" fromto="0 -0.15 0 0 0.15 0" size="0.03" friction="0.5 0.1 0.1" mass="0.5"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m)

  two_rail_contacts = 0
  for step in range(80):
    sim.step(1)
    mujoco.mj_step(m, d_cpu)

    a = sim.assembled_system()
    ncon = int(np.sum(a["contact_mask"][0].cpu().numpy() > 0.5))
    if ncon == 2:
      two_rail_contacts += 1

  assert sim.state.status[0].item() == 0
  q_gpu = sim.state.qpos[0].cpu().numpy()
  np.testing.assert_allclose(q_gpu, d_cpu.qpos, rtol=1e-3, atol=1e-4)
  assert two_rail_contacts == 80, f"Expected continuous 2-rail contact across all 80 steps, got {two_rail_contacts}"


# =============================================================================
# 4. Explicit Pairs, Body Exclusions, and Filtering
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_explicit_pairs_exclusions_and_filtering():
  """Verifies explicit pair parameters and body exclusion suppressions with active contacts."""
  xml = """<mujoco>
  <option timestep="0.002" iterations="500" tolerance="1e-5"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b1" pos="0 0 0.08">
      <joint type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size="0.1"/>
    </body>
    <body name="b2" pos="0 0 0.15">
      <joint type="slide" axis="0 0 1"/>
      <geom name="g2" type="sphere" size="0.1"/>
    </body>
    <body name="b3" pos="0.5 0 0.08">
      <joint type="slide" axis="0 0 1"/>
      <geom name="g3" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <contact>
    <!-- Explicit pair with custom anisotropic friction and condim=3 -->
    <pair geom1="floor" geom2="g1" friction="0.9 0.4 0.01" condim="3"/>
    <!-- Body exclude between b1 and b2 -->
    <exclude body1="b1" body2="b2"/>
  </contact>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  desc = lower_coupled_constraints(m)

  # 1. Total 4 geoms: 6 pairs minus (b1, b2) excluded = 5 candidate pairs
  assert desc.npairs == 5, f"Expected 5 candidate pairs, got {desc.npairs}"
  pair_names = [(m.geom(desc.geom1[p]).name, m.geom(desc.geom2[p]).name) for p in range(desc.npairs)]
  assert ("g1", "g2") not in pair_names and ("g2", "g1") not in pair_names

  # 2. Check anisotropic friction parameters
  assert desc.contact_friction[0, 0] == 0.9
  assert desc.contact_friction[0, 1] == 0.4

  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  # 3. Verify native contact mask only activates floor-g1 and b3 contacts, and b1-b2 is suppressed
  mask = coupled["contact_mask"][0].cpu().numpy()
  assert mask[0] == 1.0, "Explicit floor-g1 contact must be active"

  sim.step(1)
  assert sim.state.status[0].item() == 0


# =============================================================================
# 5. GPU Capacity Boundaries, Final Slots, and Overflow Rejection
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_gpu_capacity_boundaries_execution():
  """Execute accepted capacity boundaries nv=24, ncontacts_max=24, nr=96 on GPU with direct CPU comparison."""
  # 3 box-box pairs (3 * 8 = 24 contact slots) with condim=1 -> 24 contact rows
  # 24 slide joints (nv=24, njnt=24) with limited=true and frictionloss > 0 -> 24 friction + 48 limit = 72 joint rows
  # Total nr = 72 + 24 = 96 rows!
  xml_nc24 = """<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-4"/><worldbody>"""
  for i in range(3):
    xml_nc24 += f"""
    <body name="b1_{i}" pos="{i*2} 0 0"><geom name="g1_{i}" type="box" size="0.1 0.1 0.05" conaffinity="0"/></body>
    <body name="b2_{i}" pos="{i*2} 0 0.08" quat="0.9238795 0 0 0.3826834">
      <joint name="jb_{i}" type="slide" axis="0 0 1" range="-1 1" limited="true" frictionloss="0.01"/>
      <geom name="g2_{i}" type="box" size="0.08 0.08 0.04" conaffinity="0"/>
    </body>
    """
  for i in range(21):
    xml_nc24 += f"""<body pos="0 {i} 10"><joint name="j{i}" type="slide" axis="0 0 1" range="-1 1" limited="true" frictionloss="0.01"/><geom type="sphere" size="0.01" conaffinity="0"/></body>"""
  xml_nc24 += "</worldbody><contact>"
  for i in range(3):
    xml_nc24 += f'<pair geom1="g1_{i}" geom2="g2_{i}" condim="1"/>'
  xml_nc24 += "</contact></mujoco>"

  m96 = mujoco.MjModel.from_xml_string(xml_nc24)
  desc = lower_coupled_constraints(m96)
  assert desc.npairs == 3
  assert desc.ncontacts_max == 24
  assert desc.nr == 96
  assert m96.nv == 24

  d_cpu = mujoco.MjData(m96)
  mujoco.mj_forward(m96, d_cpu)
  assert d_cpu.ncon == 24

  sim96 = MetalSimulation(m96, batch_size=1, profile="integrated_euler_v1")
  assembled96 = sim96.assembled_system()
  sim96.step(1)

  assert sim96.state.status[0].item() == 0

  # Assert contact slot 23 (the 24th slot) is active
  mask = assembled96["contact_mask"][0].cpu().numpy()
  assert np.sum(mask > 0.5) == 24, "All 24 contact slots must be active"
  assert mask[23] > 0.5, "Contact slot 23 must be active"

  # Assert constraint row slot 95 (the 96th row) is active
  J96 = assembled96["J"][0].cpu().numpy()
  assert np.linalg.norm(J96[95]) > 0, "Row slot 95 must be active in J"

  # Direct CPU comparison
  qacc_gpu = sim96.state.qacc[0].cpu().numpy()
  np.testing.assert_allclose(qacc_gpu, d_cpu.qacc, rtol=1e-3, atol=2e-4)


def test_capacity_overflow_rejection():
  """Admission guards cleanly reject models exceeding pair, contact, or row limits before construction."""
  # 1. 17 pairs exceeds capacity 16
  xml_p17 = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>"""
  for i in range(17):
    xml_p17 += f"""<body pos="{i} 0 1"><geom name="s{i}" type="sphere" size=".1" conaffinity="0"/></body>"""
  xml_p17 += """</worldbody><contact>"""
  for i in range(17):
    xml_p17 += f"""<pair geom1="floor" geom2="s{i}"/>"""
  xml_p17 += """</contact></mujoco>"""

  with pytest.raises(ValueError, match="candidate contact pairs \\(17\\) exceeds capacity 16"):
    lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_p17))

  # 2. 25 contact slots exceeds capacity 24 (3 box-box pairs = 24 slots, plus 1 sphere-sphere pair = 25 slots)
  xml_c25 = """<mujoco><worldbody>"""
  for i in range(3):
    xml_c25 += f"""<body pos="{i*2} 0 0"><geom name="b1_{i}" type="box" size=".1 .1 .1"/></body>
    <body pos="{i*2} 0 1"><geom name="b2_{i}" type="box" size=".1 .1 .1"/></body>"""
  xml_c25 += """<body pos="10 0 0"><geom name="s1" type="sphere" size=".1"/></body>
  <body pos="10 0 1"><geom name="s2" type="sphere" size=".1"/></body>"""
  xml_c25 += """</worldbody><contact>"""
  for i in range(3):
    xml_c25 += f"""<pair geom1="b1_{i}" geom2="b2_{i}"/>"""
  xml_c25 += """<pair geom1="s1" geom2="s2"/>"""
  xml_c25 += """</contact></mujoco>"""

  with pytest.raises(ValueError, match="total candidate contact slots \\(25\\) exceeds capacity 24"):
    lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_c25))

  # 3. 99 rows exceeds capacity 96 (3 box-box pairs with condim=1 = 24 rows, plus 25 slide joints = 75 rows -> 99 rows)
  xml_r97 = """<mujoco><compiler angle="radian"/><worldbody>"""
  for i in range(3):
    xml_r97 += f"""<body pos="{i*2} 0 0"><geom name="b1_{i}" type="box" size=".1 .1 .1" conaffinity="0"/></body>
    <body pos="{i*2} 0 1"><geom name="b2_{i}" type="box" size=".1 .1 .1" conaffinity="0"/></body>"""
  for i in range(25):
    xml_r97 += f"""<body pos="0 {i} 10"><joint name="j{i}" type="slide" axis="0 0 1" range="-1 1" limited="true" frictionloss="0.01"/><geom type="sphere" size="0.01" conaffinity="0"/></body>"""
  xml_r97 += """</worldbody><contact>"""
  for i in range(3):
    xml_r97 += f"""<pair geom1="b1_{i}" geom2="b2_{i}" condim="1"/>"""
  xml_r97 += """</contact></mujoco>"""

  with pytest.raises(ValueError, match="total candidate constraint rows \\(99\\) exceeds capacity 96"):
    lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_r97))


# =============================================================================
# 6. Checkpoint Snapshot, Restore, Replay & Failure Isolation
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_checkpoint_snapshot_restore_and_replay():
  """Snapshot / restore / replay with verified active contact manifold, separation, re-impact, and isolation."""
  import torch

  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-5"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.049">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.05" friction="0.8 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)

  q0 = np.zeros((2, 7), dtype=np.float32)
  q0[0] = [0, 0, 0.049, 1, 0, 0, 0]
  q0[1] = [0, 0, 0.049, 1, 0, 0, 0]

  v0 = np.zeros((2, 6), dtype=np.float32)
  v0[0, 2] = 0.5  # World 0 launches upward
  v0[1, 2] = 0.0  # World 1 remains resting

  sim = MetalSimulation(m, batch_size=2, qpos=q0, qvel=v0, profile="integrated_euler_v1")

  # 1. Verify initial active contact manifold on both worlds (4-corner manifold)
  a0 = sim.assembled_system()
  mask0 = a0["contact_mask"].cpu().numpy()
  assert np.sum(mask0[0] > 0.5) == 4, "World 0 must start with active 4-corner contact manifold"
  assert np.sum(mask0[1] > 0.5) == 4, "World 1 must start with active 4-corner contact manifold"

  # 2. Step 10 steps: World 0 separates (contacts clear, ncon -> 0), World 1 stays in contact (ncon == 4)
  for _ in range(10):
    sim.step(1)

  a_mid = sim.assembled_system()
  mask_mid = a_mid["contact_mask"].cpu().numpy()
  assert np.sum(mask_mid[0] > 0.5) == 0, "World 0 contacts must clear during airborne flight"
  assert np.sum(mask_mid[1] > 0.5) == 4, "World 1 must maintain active 4-corner contact manifold"

  # 3. Snapshot during mixed active/separated state
  chk = sim.state.snapshot()

  # 4. Step 45 more steps: World 0 re-impacts ground (ncon -> 4)
  for _ in range(45):
    sim.step(1)

  a_end = sim.assembled_system()
  mask_end = a_end["contact_mask"].cpu().numpy()
  assert np.sum(mask_end[0] > 0.5) == 4, "World 0 must re-impact and re-establish 4 contacts"
  assert np.sum(mask_end[1] > 0.5) == 4, "World 1 must maintain 4 contacts"

  qpos_target = sim.state.qpos.clone()
  qvel_target = sim.state.qvel.clone()
  qacc_target = sim.state.qacc.clone()

  # 5. Test selective reset isolation: reset World 1, World 0 unaffected
  qpos_before = sim.state.qpos.clone()
  sim.state.reset(env_ids=[1])
  assert torch.equal(sim.state.qpos[0], qpos_before[0]), "Selective reset of World 1 must isolate World 0"

  # 6. Test restore and replay
  sim.state.restore(chk)
  for _ in range(45):
    sim.step(1)

  assert torch.equal(sim.state.qpos, qpos_target), "Bit-for-bit qpos replay after restore"
  assert torch.equal(sim.state.qvel, qvel_target), "Bit-for-bit qvel replay after restore"
  assert torch.equal(sim.state.qacc, qacc_target), "Bit-for-bit qacc replay after restore"


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_clockwork_parcel_sorter_demo_check():
  """Clockwork parcel sorter demo runs headless check with real physical routing."""
  import sys
  from pathlib import Path
  examples_dir = Path(__file__).resolve().parent.parent / "examples"
  if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))
  from clockwork_parcel_sorter import run

  metrics = run(steps=500, mode="metal", check=True)
  assert metrics["box_routed_left"] is True
  assert metrics["cap_routed_right"] is True
  assert metrics["sph_routed_center"] is True
  assert metrics["active_contact_steps"] >= 200
  assert metrics["native_active_contact_steps"] >= 200
  assert metrics["peak_contacts"] >= 3
  assert metrics["native_peak_contacts"] >= 3
  assert len(metrics["unique_contact_pairs"]) >= 4
  assert metrics["max_stage_sensor_error"] <= 1e-6
  assert metrics["max_early_qpos_error"] <= 5e-5
  assert metrics["max_full_qpos_error"] <= 0.15

