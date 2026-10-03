# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit inverse-query storage ownership and rollback regressions."""
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize("fail", [False, True])
def test_inverse_query_preserves_nested_borrowed_stages_on_success_or_failure(fail):
  torch = pytest.importorskip("torch")
  from mujoco_metal.native_api import _inverse_query_workspaces

  class Stage:
    pass
  # Use the same subordinate-program discovery contract as real FK stages.
  Stage.__module__ = "mujoco_metal.test_query_stage"
  fk, smooth = Stage(), Stage()
  fk._workspace = {"outputs": {"body_pos": torch.tensor([[1., 2., 3.]])}}
  smooth._fk = fk
  smooth._workspace = {"mass": torch.tensor([[[2., .3], [.3, 1.]]])}
  smooth._cache_epoch = 4
  fk_pos = fk._workspace["outputs"]["body_pos"]
  mass = smooth._workspace["mass"]
  saved_pos, saved_mass = fk_pos.clone(), mass.clone()
  sim = SimpleNamespace(_smooth=smooth, _spatial_kin=None, _spatial_cache_key=(1, 2))
  def query():
    with _inverse_query_workspaces(sim):
      mass.mul_(7)
      fk_pos.add_(3)
      smooth._workspace["mass"] = torch.zeros_like(mass)
      smooth._workspace["new_view"] = fk_pos
      smooth._cache_epoch = 9
      fk._workspace = {"outputs": {"body_pos": torch.ones_like(fk_pos)}}
      sim._spatial_kin, sim._spatial_cache_key = {}, None
      if fail:
        raise RuntimeError("temporary forward failed")
  if fail:
    with pytest.raises(RuntimeError, match="temporary forward failed"):
      query()
  else:
    query()
  assert smooth._workspace["mass"] is mass
  assert fk._workspace["outputs"]["body_pos"] is fk_pos
  assert set(smooth._workspace) == {"mass"}
  assert smooth._cache_epoch == 4
  torch.testing.assert_close(mass, saved_mass, rtol=0, atol=0)
  torch.testing.assert_close(fk_pos, saved_pos, rtol=0, atol=0)
  assert sim._spatial_kin is None
  assert sim._spatial_cache_key == (1, 2)
