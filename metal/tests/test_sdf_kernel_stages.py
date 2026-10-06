"""CPU source contracts and opt-in native selectors for staged SDF kernels."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest


_GPU = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native Metal")
_ROOT = Path(__file__).resolve().parents[1]
_VBOX = ("-0.05 -0.05 -0.04 0.05 -0.05 -0.04 "
         "0.05 0.05 -0.04 -0.05 0.05 -0.04 "
         "-0.05 -0.05 0.04 0.05 -0.05 0.04 "
         "0.05 0.05 0.04 -0.05 0.05 0.04")


def _scene(initpoints, z=0.08):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><asset><mesh name="box" vertex="{_VBOX}"/></asset>'
      f'<option timestep="0.002" integrator="Euler" iterations="40" '
      f'tolerance="1e-8" gravity="0 0 0" sdf_initpoints="{initpoints}"/>'
      f'<worldbody><geom name="sdf" type="sdf" mesh="box" condim="1" '
      f'contype="1" conaffinity="1"/><body pos="0 0 {z}"><freejoint/>'
      f'<geom name="ball" type="sphere" size="0.05" condim="1" '
      f'contype="1" conaffinity="1"/></body></worldbody></mujoco>')


@pytest.mark.parametrize("budget, contacts", [(4, 4), (40, 34), (51, 44)])
def test_active_cpu_sdf_allows_more_seeds_than_output_slots(budget, contacts):
  """Active source callback works above 50 seeds when accepted output stays <=50."""
  model = _scene(budget)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == contacts <= 50


def test_sdf_stage_workspace_is_bounded_and_all_seeds_are_host_dispatched():
  host = (_ROOT / "mujoco_metal" / "coupled_constraints.py").read_text()
  shader = (_ROOT / "mujoco_metal" / "shaders" / "coupled_constraints.metal").read_text()
  narrow = (_ROOT / "mujoco_metal" / "shaders" / "sdf_narrowphase.metal").read_text()
  for name in ("contact_sdf_seed_init", "contact_sdf_phase_reset",
               "contact_sdf_descent_prepare", "contact_sdf_line_search",
               "contact_sdf_publish_normal", "contact_sdf_publish_contact"):
    assert name in shader
    assert name in host
  assert "for step in range(self._sdf_max_iterations)" in host
  assert "range(0, self._sdf_seed_count, self._sdf_seed_tile)" in host
  assert "SDF_STAGE_STATE_WORDS 159" in narrow
  assert "_SDF_STAGE_STATE_WORDS = 159" in host
  assert "_contact_sdf_seed_producer" not in host
  from mujoco_metal.coupled_constraints import _pair_contact_cache_bytes
  assert _pair_contact_cache_bytes(1, 50, 1, 32) == 31232
  assert _pair_contact_cache_bytes(1, 0, 0, 0) == 48


def test_cpu_active_51_seed_pair_gets_50_output_slots():
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  descriptor = lower_coupled_constraints(
      _scene(51), limits=CapacityLimits(max_slots=50))
  assert descriptor.ncontacts_max == 50
  assert int(descriptor.pair_max_contacts.max()) == 50


@_GPU
def test_native_staged_sdf_contact_selector():
  from mujoco_metal.simulation import MetalSimulation

  model = _scene(4)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  result = sim.assembled_system(recompute=True)
  assert int(sim._coupled_constraints._workspace["contact_pair_count"].sum().item()) == 4
  assert int(result["status"].detach().cpu().numpy().reshape(-1)[0]) == 0


@_GPU
def test_native_staged_51_seed_44_source_contacts():
  """Exercise all 51 configured seeds against pinned CPU's 44 contacts."""
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.simulation import MetalSimulation
  from test_sdf_contact_013 import _check

  model = _scene(51)
  source = mujoco.MjData(model)
  mujoco.mj_forward(model, source)
  cpu = [(float(source.contact[i].dist),
          np.asarray(source.contact[i].frame[:3]),
          np.asarray(source.contact[i].pos)) for i in range(source.ncon)]
  assert len(cpu) == 44

  limits = CapacityLimits(max_slots=50, max_rows=256)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1",
                        limits=limits)
  native = sim.assembled_system(recompute=True)
  mask = native["contact_mask"].detach().cpu().numpy()[0]
  distance = native["contact_distance"].detach().cpu().numpy()[0]
  normal = native["contact_normal"].detach().cpu().numpy()[0]
  position = native["contact_position"].detach().cpu().numpy()[0]
  contacts = [(float(distance[i]), normal[i], position[i])
              for i in range(len(mask)) if mask[i] > 0.5]
  assert len(contacts) == 44
  _check("staged-51-seed", "press", cpu, contacts)


@_GPU
def test_native_staged_sdf_zero_and_large_seed_selectors():
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  zero = MetalCoupledConstraints(_scene(0, z=10.0), batch_size=1)
  assert zero.descriptor.ncontacts_max == 0
  assert zero._contact_sdf_seed_init is not None
  large = MetalCoupledConstraints(
      _scene(51, z=0.08), batch_size=1,
      limits=CapacityLimits(max_slots=50))
  assert large._sdf_seed_count == 51
  assert large.descriptor.ncontacts_max == 50
  assert large._workspace["contact_sdf_seed_state"].numel() == (
      large.descriptor.npairs * 32 * 159)
