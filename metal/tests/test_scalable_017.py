# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 017: Scalable device execution qualification (R08).

Covers:
- Block-row solving beyond the 96-row threshold (nr > 96, nv <= 64).
- Candidate compaction and telemetry (work_ratio, active vs allocated counts).
- Exact capacity limit enforcement and overflow diagnostics.
- Kinematic island discovery, sleep/wake lifecycle, and mjDSBL_ISLAND semantics.
- Multi-world batch isolation (healthy neighbor guarantee).
"""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
from mujoco_metal.islands import build_island_partition, IslandManager

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
    ),
]


def test_capacity_limits_exact_overflow_reporting_cpu():
  """Exact overflow reporting when limits are exceeded."""
  spheres = []
  for i in range(4):
    x = -0.6 + 0.3 * i
    spheres.append(f"""
      <body name="b_{i}" pos="{x} 0 0.08">
        <joint name="j_{i}_x" type="slide" axis="1 0 0" range="-1 1"/>
        <joint name="j_{i}_y" type="slide" axis="0 1 0" range="-1 1"/>
        <joint name="j_{i}_z" type="slide" axis="0 0 1" range="-1 1"/>
        <geom name="g_{i}" type="sphere" size="0.08" condim="6"/>
      </body>
    """)
  xml = f"""
  <mujoco>
    <option timestep="0.002" cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      {"".join(spheres)}
    </worldbody>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  # Check that custom limit with tight max_rows raises CapacityOverflow with exact text
  tight = CapacityLimits(max_rows=50, max_slots=64, max_pairs=64, max_nv=64)
  with pytest.raises(CapacityOverflow, match=r"candidate constraint rows \(136\) exceeds capacity 50"):
    lower_coupled_constraints(m, limits=tight)

  # Check that tight max_nv raises CapacityOverflow with exact text
  tight_nv = CapacityLimits(max_rows=256, max_slots=64, max_pairs=64, max_nv=10)
  with pytest.raises(CapacityOverflow, match="bounds nv to 10"):
    lower_coupled_constraints(m, limits=tight_nv)


def test_scalable_block_row_solving_nr_gt_96_gpu():
  """Block-row solver solves nr > 96 models on GPU and matches CPU MuJoCo oracle."""
  from mujoco_metal.simulation import MetalSimulation
  # 4 spheres with condim=6 and slide joint limits produce nr=136 rows, nv=12 DOFs
  spheres = []
  for i in range(4):
    x = -0.6 + 0.3 * i
    spheres.append(f"""
      <body name="b_{i}" pos="{x} 0 0.08">
        <joint name="j_{i}_x" type="slide" axis="1 0 0" range="-1 1"/>
        <joint name="j_{i}_y" type="slide" axis="0 1 0" range="-1 1"/>
        <joint name="j_{i}_z" type="slide" axis="0 0 1" range="-1 1"/>
        <geom name="g_{i}" type="sphere" size="0.08" condim="6"/>
      </body>
    """)

  xml = f"""
  <mujoco>
    <option timestep="0.002" cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      {"".join(spheres)}
    </worldbody>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  limits = CapacityLimits(max_rows=192, max_slots=64, max_pairs=64, max_nv=64)

  sim = MetalSimulation(m, batch_size=1, profile="integrated_scalable_v1", limits=limits)
  cc = sim._coupled_constraints
  assert cc.descriptor.nr == 136, f"Expected nr=136, got {cc.descriptor.nr}"
  assert not cc.descriptor.dense_path, "Expected dense_path == False for nr > 96"

  # Reset and step
  d_cpu = mujoco.MjData(m)
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))

  for step_idx in range(15):
    sim.step(1)
    mujoco.mj_step(m, d_cpu)

    # Check status
    assert sim.state.status.cpu().numpy()[0] == 0, f"Solver failed at step {step_idx}"

  # Check numerical parity against CPU oracle
  q_gpu = sim.state.qpos.cpu().numpy()[0]
  q_cpu = np.asarray(d_cpu.qpos)
  np.testing.assert_allclose(q_gpu, q_cpu, atol=1e-5, rtol=1e-4)

  v_gpu = sim.state.qvel.cpu().numpy()[0]
  v_cpu = np.asarray(d_cpu.qvel)
  np.testing.assert_allclose(v_gpu, v_cpu, atol=1e-5, rtol=1e-4)


def test_candidate_compaction_and_work_telemetry_gpu():
  """Candidate compaction metrics report accurate active vs allocated work."""
  from mujoco_metal.simulation import MetalSimulation
  # One sphere touching floor, one sphere far away in the air
  xml = """
  <mujoco>
    <option timestep="0.002" cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <body name="near" pos="0 0 0.08">
        <joint type="free"/>
        <geom type="sphere" size="0.08" condim="3"/>
      </body>
      <body name="far" pos="0 0 10.0">
        <joint type="free"/>
        <geom type="sphere" size="0.08" condim="3"/>
      </body>
    </worldbody>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_scalable_v1")
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  sim.step(1)

  cc = sim._coupled_constraints
  metrics = cc.candidate_compaction_metrics()[0]
  assert "work_ratio" in metrics
  assert "active_slots" in metrics
  assert "allocated_slots" in metrics
  assert metrics["work_ratio"] < 1.0  # far pair does not do contact solve work
  assert metrics["active_slots"] < metrics["allocated_slots"]


def test_island_discovery_and_sleep_wake_gpu():
  """Kinematic island discovery and sleep/wake lifecycle."""
  xml = """
  <mujoco>
    <option timestep="0.002" gravity="0 0 0">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <body name="box1" pos="-1 0 0.1">
        <joint name="j1" type="slide" axis="0 0 1"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
      <body name="box2" pos="1 0 0.1">
        <joint name="j2" type="slide" axis="0 0 1"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  partition = build_island_partition(m)
  assert partition.ntree == 2
  assert len(partition.tree_dofs[0]) == 1
  assert len(partition.tree_dofs[1]) == 1

  from mujoco_metal.simulation import MetalSimulation
  sim = MetalSimulation(m, batch_size=1, profile="integrated_scalable_v1")
  sim.islands.sleep_delay_steps = 3
  sim.reset()

  # Step enough times for stationary bodies to fall asleep
  for _ in range(5):
    sim.step(1)

  assert bool(sim.islands.tree_asleep[0, 0])
  assert bool(sim.islands.tree_asleep[0, 1])

  # Perturb tree 0 by setting qvel
  sim._state._qvel[0, 0] = 0.5
  sim.step(1)

  # Tree 0 should be awake, tree 1 remains asleep
  assert not bool(sim.islands.tree_asleep[0, 0])
  assert bool(sim.islands.tree_asleep[0, 1])

  # Wake all
  sim.islands.wake_all()
  assert not bool(sim.islands.tree_asleep[0, 0])
  assert not bool(sim.islands.tree_asleep[0, 1])


def test_mjdsbl_island_gpu():
  """mjDSBL_ISLAND disables island decomposition."""
  xml = """
  <mujoco>
    <option timestep="0.002">
      <flag sleep="enable" island="disable"/>
    </option>
    <worldbody>
      <body name="b1" pos="-1 0 0.5"><joint type="slide"/><geom type="sphere" size="0.1"/></body>
      <body name="b2" pos="1 0 0.5"><joint type="slide"/><geom type="sphere" size="0.1"/></body>
    </worldbody>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  mgr = IslandManager(m, batch_size=1)
  assert mgr.island_disabled
  islands = mgr.discover_islands()
  assert len(islands) == 1  # single island grouping all trees


def test_healthy_neighbor_isolation_gpu():
  """Healthy worlds continue advancing even if neighbor world diverges."""
  from mujoco_metal.simulation import MetalSimulation
  xml = """
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <body name="b" pos="0 0 0.5">
        <joint type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_scalable_v1")
  sim.reset()

  # Step 5 times healthy
  sim.step(5)
  assert np.all(sim.state.status.cpu().numpy() == 0)

  # Induce failure in world 0 only (set qpos NaN)
  sim._state._qpos[0, 0] = float("nan")
  sim.step(1)

  status = sim.state.status.cpu().numpy()
  # World 0 should report failure
  assert status[0] != 0
  # World 1 must remain healthy (status 0)
  assert status[1] == 0
