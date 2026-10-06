"""CPU contract tests for pre-mutation coupled pose validation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

METAL = Path(__file__).resolve().parents[1]
SOURCE = METAL / "mujoco_metal" / "coupled_constraints.py"
_SPEC = importlib.util.spec_from_file_location(
    "mujoco_metal._coupled_pose_candidate", SOURCE)
_COUPLED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COUPLED)

FIELDS = {
    "geom_xmat": (2, 3, 9),
    "geom_xmat_low": (2, 3, 9),
    "geom_xmat_tail": (2, 3, 9),
    "geom_quat_low": (2, 3, 4),
    "geom_quat_tail": (2, 3, 4),
}


def _poses():
  result = {name: torch.zeros(shape) for name, shape in FIELDS.items()}
  result.update(body_pos=torch.zeros((2, 2, 3)),
                body_quat=torch.zeros((2, 2, 4)))
  return result


def _solver_stub():
  return SimpleNamespace(
      _torch=torch,
      _device=torch.device("cpu"),
      descriptor=SimpleNamespace(ngeom=3, nbody=2, nv=1, nq=1,
                                 nr=1, ncontacts_max=1, neq=0,
                                 dense_path=True),
      batch_size=2,
      _workspace={"sentinel": torch.full((2,), 19.)},
      _candidate_generation_token=object(),
      _position_current_valid=True,
      _position_context_valid=True,
      _position_context_epoch=5,
      _debug_input=torch.full((2,), 23.),
      _status=torch.full((2,), 29, dtype=torch.int32),
      _cache=torch.full((2, 2), 31.),
      _kernel_calls=0,
  )


@pytest.mark.parametrize("field", tuple(FIELDS))
@pytest.mark.parametrize("fault", ("missing", "dtype", "shape", "device", "layout"))
def test_geom_rotation_pose_abi_rejects_malformed_real_tensors(field, fault):
  poses = _poses()
  if fault == "missing":
    poses.pop(field)
  elif fault == "dtype":
    poses[field] = torch.zeros(FIELDS[field], dtype=torch.float64)
  elif fault == "shape":
    poses[field] = torch.zeros((2, 3, 10 if field.startswith("geom_xmat") else 5))
  elif fault == "device":
    poses[field] = torch.empty(FIELDS[field], device="meta")
  else:
    width = FIELDS[field][-1]
    poses[field] = torch.zeros((*FIELDS[field][:-1], width * 2))[..., ::2]
  with pytest.raises(ValueError, match=field):
    _COUPLED._validate_geom_pose_residuals(torch, torch.device("cpu"),
                                           poses, 2, 3)


@pytest.mark.parametrize("entry", ("run_device", "run_velocity_device"))
@pytest.mark.parametrize("field", tuple(FIELDS))
@pytest.mark.parametrize("fault", ("missing", "type", "shape", "dtype",
                                    "device", "layout"))
def test_public_solver_entries_reject_bad_pose_before_any_state_mutation(
    entry, field, fault):
  solver = _solver_stub()
  token = solver._candidate_generation_token
  workspace = {key: value.clone() for key, value in solver._workspace.items()}
  debug = solver._debug_input.clone()
  status = solver._status.clone()
  cache = solver._cache.clone()
  poses = _poses()
  if fault == "missing":
    poses.pop(field)
  elif fault == "type":
    poses[field] = object()
  elif fault == "shape":
    poses[field] = torch.zeros((2, 3, 10 if field.startswith("geom_xmat") else 5))
  elif fault == "dtype":
    poses[field] = torch.zeros(FIELDS[field], dtype=torch.float64)
  elif fault == "device":
    poses[field] = torch.empty(FIELDS[field], device="meta")
  else:
    width = FIELDS[field][-1]
    poses[field] = torch.zeros((*FIELDS[field][:-1], width * 2))[..., ::2]
  if entry == "run_device":
    with pytest.raises(ValueError, match=field):
      _COUPLED.MetalCoupledConstraints.run_device(
          solver, poses, torch.ones((2, 1, 1)), None, None, None)
  else:
    context = {"_owner": solver, "_epoch": 5}
    with pytest.raises(ValueError, match=field):
      _COUPLED.MetalCoupledConstraints.run_velocity_device(
          solver, context, poses, None, None, None, None)
  assert solver._candidate_generation_token is token
  assert solver._position_current_valid is True
  assert solver._position_context_valid is True
  assert solver._position_context_epoch == 5
  assert solver._kernel_calls == 0
  torch.testing.assert_close(solver._debug_input, debug, rtol=0, atol=0)
  torch.testing.assert_close(solver._status, status, rtol=0, atol=0)
  torch.testing.assert_close(solver._cache, cache, rtol=0, atol=0)
  for key, value in workspace.items():
    torch.testing.assert_close(solver._workspace[key], value, rtol=0, atol=0)
