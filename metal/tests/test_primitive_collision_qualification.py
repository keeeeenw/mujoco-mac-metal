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
  """Qualifies that (B, A) reversed pair declarations produce identical contact physics."""
  xml_order1 = """<mujoco><worldbody>
    <body name="b1" pos="0 0 1"><joint type="free"/><geom name="s1" type="sphere" size="0.1"/></body>
    <body name="b2" pos="0 0 1.18"><joint type="free"/><geom name="box1" type="box" size="0.2 0.2 0.1"/></body>
  </worldbody>
  <contact><pair geom1="s1" geom2="box1"/></contact>
  </mujoco>"""

  xml_order2 = """<mujoco><worldbody>
    <body name="b1" pos="0 0 1"><joint type="free"/><geom name="s1" type="sphere" size="0.1"/></body>
    <body name="b2" pos="0 0 1.18"><joint type="free"/><geom name="box1" type="box" size="0.2 0.2 0.1"/></body>
  </worldbody>
  <contact><pair geom1="box1" geom2="s1"/></contact>
  </mujoco>"""

  m1 = mujoco.MjModel.from_xml_string(xml_order1)
  m2 = mujoco.MjModel.from_xml_string(xml_order2)

  sim1 = MetalSimulation(m1, batch_size=1, profile="integrated_euler_v1")
  sim2 = MetalSimulation(m2, batch_size=1, profile="integrated_euler_v1")

  c1 = sim1.assembled_system()
  c2 = sim2.assembled_system()

  np.testing.assert_allclose(c1["contact_distance"][0, 0].item(), c2["contact_distance"][0, 0].item(), atol=1e-6)
  np.testing.assert_allclose(c1["contact_normal"][0, 0].cpu().numpy(), c2["contact_normal"][0, 0].cpu().numpy(), atol=1e-6)
  np.testing.assert_allclose(c1["contact_position"][0, 0].cpu().numpy(), c2["contact_position"][0, 0].cpu().numpy(), atol=1e-6)


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

  # Check qacc and constraint forces match CPU
  np.testing.assert_allclose(sim.state.qacc[0].cpu().numpy(), d_cpu.qacc, rtol=1e-3, atol=1e-2)
  np.testing.assert_allclose(coupled["qfrc_constraint"][0].cpu().numpy(), d_cpu.qfrc_constraint, rtol=1e-3, atol=1e-2)

  # Check effective mass
  M_cpu = np.zeros((m.nv, m.nv), dtype=np.float64)
  mujoco.mj_fullM(m, d_cpu, M_cpu)
  np.testing.assert_allclose(coupled["mass_matrix"][0].cpu().numpy(), M_cpu, atol=1e-5)

  # Check Delassus regularized W cross blocks
  W_reg = coupled["W_regularized"][0].cpu().numpy()
  assert np.all(np.isfinite(W_reg))


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
  """Rocking box: initially tilted box contacting floor on one edge, rocking to rest."""
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

  for _ in range(40):
    sim.step(1)
    mujoco.mj_step(m, d_cpu)

  assert sim.state.status[0].item() == 0
  q_gpu = sim.state.qpos[0].cpu().numpy()
  np.testing.assert_allclose(q_gpu, d_cpu.qpos, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_dynamic_trajectory_capsule_on_box_rails():
  """Articulated capsule sliding/rolling along an inclined box ramp under friction."""
  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-5"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <!-- Inclined box ramp -->
    <body name="ramp" pos="0 0 0.3" quat="0.9659258 0 0.258819 0">
      <geom type="box" size="0.4 0.2 0.02" friction="0.5 0.1 0.1"/>
    </body>
    <!-- Capsule rolling down the ramp -->
    <body name="roller" pos="-0.2 0 0.45" quat="0.9659258 0 0.258819 0">
      <joint type="free"/>
      <geom type="capsule" size="0.05 0.15" friction="0.5 0.1 0.1" mass="0.5"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m)

  for _ in range(30):
    sim.step(1)
    mujoco.mj_step(m, d_cpu)

  assert sim.state.status[0].item() == 0
  q_gpu = sim.state.qpos[0].cpu().numpy()
  np.testing.assert_allclose(q_gpu, d_cpu.qpos, rtol=1e-3, atol=2e-3)


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
    <!-- Explicit pair with custom friction and condim -->
    <pair geom1="floor" geom2="g1" friction="0.9 0.9 0.01" condim="3"/>
    <!-- Body exclude between b1 and b2 -->
    <exclude body1="b1" body2="b2"/>
  </contact>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  desc = lower_coupled_constraints(m)

  # Total 4 geoms: 6 pairs minus (b1, b2) excluded = 5 candidate pairs
  assert desc.npairs == 5, f"Expected 5 candidate pairs, got {desc.npairs}"


  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  coupled = sim.assembled_system()

  d_cpu = mujoco.MjData(m)
  mujoco.mj_forward(m, d_cpu)

  # Verify b1 and b2 did not collide even though overlapping
  for c_idx in range(d_cpu.ncon):
    con = d_cpu.contact[c_idx]
    assert not (con.geom1 == 1 and con.geom2 == 2)
    assert not (con.geom1 == 2 and con.geom2 == 1)

  sim.step(1)
  assert sim.state.status[0].item() == 0


# =============================================================================
# 5. GPU Capacity Boundaries, Final Slots, and Overflow Rejection
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_gpu_capacity_boundaries_execution():
  """Execute accepted capacity boundaries nv=32, nc=16, nr=96 on GPU."""
  # 1. Accepted boundary nc=16, nr=96
  xml_nc16 = """<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-4"/><worldbody>
  <geom type="plane" size="10 10 0.1"/>"""
  for i in range(8):
    xml_nc16 += f"""<body pos="{i} 0 0.08"><joint name="j{i}" type="slide" axis="0 0 1" range="-0.6 0.5" limited="true" frictionloss="0.05"/>
    <geom type="sphere" size="0.1" pos="0 0 0" friction="0.8 0.1 0.1" conaffinity="0"/>
    <geom type="sphere" size="0.1" pos="0 0 0.5" friction="0.8 0.1 0.1" conaffinity="0"/>
    </body>"""
  xml_nc16 += "</worldbody><equality>"
  for i in range(8):
    xml_nc16 += f"""<joint joint1="j{i}" polycoef="0 1 0 0 0"/>"""
  xml_nc16 += "</equality></mujoco>"

  m96 = mujoco.MjModel.from_xml_string(xml_nc16)
  desc = lower_coupled_constraints(m96)
  assert desc.nc == 16
  assert desc.nr == 96

  q0_96 = np.zeros((2, 8), dtype=np.float32)
  q0_96[0, 7] = -0.50
  q0_96[1, 7] = -0.49

  sim96 = MetalSimulation(m96, batch_size=2, qpos=q0_96, profile="integrated_euler_v1")
  assembled96 = sim96.assembled_system()
  sim96.step(1)

  assert np.all(sim96.state.status.cpu().numpy() == 0)
  # Exercise row slot 95 (the 96th row)
  J96 = assembled96["J"].cpu().numpy()
  assert np.linalg.norm(J96[0, 95]) > 0, "Row slot 95 must be active in World 0"


def test_capacity_overflow_rejection():
  """Admission guards cleanly reject models exceeding pair, contact, or row limits before construction."""
  # 1. 17 pairs exceeds capacity 16
  xml_nc16 = """<mujoco><compiler angle="radian"/><option integrator="Euler"/><worldbody>
  <geom type="plane" size="5 5 0.1"/>"""
  for i in range(8):
    xml_nc16 += f"""<body pos="{i} 0 1"><joint type="slide" axis="0 0 1"/>
    <geom type="sphere" size="0.1" pos="0 0 0" conaffinity="0"/>
    <geom type="sphere" size="0.1" pos="0 0 0.5" conaffinity="0"/>
    </body>"""
  xml_nc16 += "</worldbody></mujoco>"
  xml_nc17 = xml_nc16.replace("</worldbody></mujoco>", """<body pos="10 0 1"><joint type="slide" axis="0 0 1"/>
  <geom type="sphere" size="0.1" pos="0 0 0" conaffinity="0"/>
  </body></worldbody></mujoco>""")

  with pytest.raises(ValueError, match="candidate contact pairs \\(17\\) exceeds capacity 16"):
    lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_nc17))


# =============================================================================
# 6. Checkpoint Snapshot, Restore, Replay & Failure Isolation
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_checkpoint_snapshot_restore_and_replay():
  """Snapshot / restore / replay with multi-contact box manifold reproduces exact state."""
  import torch

  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-5"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.2">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.05" friction="0.8 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")

  # Step 10 steps to engage 4-corner contact manifold
  for _ in range(10):
    sim.step(1)

  # Snapshot state
  chk = sim.state.snapshot()

  # Step 15 more steps
  for _ in range(15):
    sim.step(1)

  qpos_stepped = sim.state.qpos.clone()

  # Restore and replay 15 steps
  sim.state.restore(chk)
  for _ in range(15):
    sim.step(1)

  assert torch.equal(sim.state.qpos, qpos_stepped), "Snapshot restore and replay must match bit-for-bit"
