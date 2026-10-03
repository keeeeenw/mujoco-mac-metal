# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU contract tests for compiled independent-tree mass block layout."""

import mujoco
import numpy as np
import pytest

from mujoco_metal.mass_layout import compile_tree_mass_layout


XML = """<mujoco><worldbody>
  <body name="a" pos="-1 0 1"><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/>
    <body pos="0 0 .4"><joint type="hinge" axis="0 1 0"/><geom type="sphere" size=".1" mass=".5"/></body>
  </body>
  <body name="b" pos="1 0 1"><freejoint/><geom type="sphere" size=".2" mass="1"/></body>
</worldbody></mujoco>"""


def test_compiled_tree_blocks_reconstruct_cpu_mass_without_cross_terms():
  model = mujoco.MjModel.from_xml_string(XML)
  layout = compile_tree_mass_layout(model.body_treeid, model.dof_bodyid)
  assert layout["ntree"] == int(model.ntree)
  assert layout["nnz"] == sum(int(n) ** 2 for n in layout["tree_dofnum"])
  assert layout["nnz"] < model.nv ** 2

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  dense = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, dense)
  blocks = np.zeros((layout["nnz"],), dtype=np.float64)
  for tree, (start, width, offset) in enumerate(zip(
      layout["tree_dofadr"], layout["tree_dofnum"],
      layout["tree_mass_offsets"])):
    start, width, offset = int(start), int(width), int(offset)
    if width == 0:
      continue
    block = dense[start:start + width, start:start + width]
    blocks[offset:offset + width * width] = block.reshape(-1)
    unrelated = np.ones(model.nv, dtype=bool)
    unrelated[start:start + width] = False
    np.testing.assert_allclose(dense[start:start + width, unrelated], 0,
                               rtol=0, atol=1e-12)
  for tree, (start, width, offset) in enumerate(zip(
      layout["tree_dofadr"], layout["tree_dofnum"],
      layout["tree_mass_offsets"])):
    start, width, offset = int(start), int(width), int(offset)
    if width:
      np.testing.assert_allclose(
          blocks[offset:offset + width * width].reshape(width, width),
          dense[start:start + width, start:start + width],
          rtol=1e-12, atol=1e-12)

  ntree = layout["ntree"]
  nv_storage = max(model.nv, 1)
  assert layout["packed"].shape == (3 * ntree + nv_storage,)
  np.testing.assert_array_equal(layout["packed"][:ntree],
                                layout["tree_dofadr"])
  np.testing.assert_array_equal(layout["packed"][ntree:2 * ntree],
                                layout["tree_dofnum"])
  np.testing.assert_array_equal(layout["packed"][2 * ntree:2 * ntree + model.nv],
                                layout["dof_treeid"])


def test_compiled_tree_layout_rejects_interleaved_tree_dofs():
  with pytest.raises(ValueError, match="contiguous block ranges"):
    compile_tree_mass_layout(np.array([-1, 0, 1], np.int32),
                             np.array([1, 2, 1], np.int32))


def test_tendon_armature_uses_pinned_mass_sparsity_for_cross_tree_terms():
  xml = """<mujoco><worldbody>
    <body name="a" pos="-1 0 0"><joint name="ja"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    <body name="b"><joint name="jb"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    <body name="c" pos="1 0 0"><joint name="jc"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody><tendon><fixed name="t" armature="1" limited="false">
      <joint joint="ja" coef="1"/><joint joint="jb" coef="1"/>
      <joint joint="jc" coef="1"/></fixed></tendon></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  assert int(model.tendon_treenum[0]) == 3
  assert model.ten_J_colind.size == 3
  layout = compile_tree_mass_layout(
      model.body_treeid, model.dof_bodyid,
      tendon_treeid=model.tendon_treeid,
      tendon_treenum=model.tendon_treenum,
      tendon_armature=model.tendon_armature,
      tendon_j_rowadr=model.ten_J_rowadr,
      tendon_j_rownnz=model.ten_J_rownnz,
      tendon_j_colind=model.ten_J_colind,
      mass_rowadr=model.M_rowadr,
      mass_rownnz=model.M_rownnz,
      mass_colind=model.M_colind)
  assert layout["ntree"] == 3
  # MuJoCo 3.10's compiled M rows contain no cross-tree columns. Its
  # mj_tendonArmature uses mju_addToSclSparseInc against that pattern, so the
  # cross-tree outer-product entries are dropped before mj_fullM expansion.
  assert layout["ncomponent"] == 3
  np.testing.assert_array_equal(layout["component_dof_ids"], np.arange(3))
  assert layout["nnz"] == model.nv

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  mass_with_armature = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass_with_armature)
  ten_j = np.zeros(model.nv, dtype=np.float64)
  adr = int(model.ten_J_rowadr[0])
  nnz = int(model.ten_J_rownnz[0])
  ten_j[np.asarray(model.ten_J_colind[adr:adr + nnz], dtype=np.int32)] = \
      np.asarray(data.ten_J[adr:adr + nnz], dtype=np.float64)
  expected = np.diag(ten_j * ten_j)
  base_model = mujoco.MjModel.from_xml_string(xml.replace('armature="1"', 'armature="0"'))
  base_data = mujoco.MjData(base_model)
  mujoco.mj_forward(base_model, base_data)
  mass_without_armature = np.zeros_like(mass_with_armature)
  mujoco.mj_fullM(base_model, base_data, mass_without_armature)
  np.testing.assert_allclose(mass_with_armature - mass_without_armature,
                             expected, rtol=2e-6, atol=2e-6)


def test_tendon_structural_layout_stays_tree_sparse_when_armature_is_zero():
  xml = """<mujoco><worldbody>
    <body name="a" pos="-1 0 0"><joint name="ja"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    <body name="b"><joint name="jb"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody><tendon><fixed name="t" limited="false">
      <joint joint="ja" coef="1"/><joint joint="jb" coef="1"/>
      </fixed></tendon></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  layout = compile_tree_mass_layout(
      model.body_treeid, model.dof_bodyid,
      tendon_treeid=model.tendon_treeid,
      tendon_treenum=model.tendon_treenum,
      tendon_armature=model.tendon_armature,
      tendon_j_rowadr=model.ten_J_rowadr,
      tendon_j_rownnz=model.ten_J_rownnz,
      tendon_j_colind=model.ten_J_colind,
      mass_rowadr=model.M_rowadr,
      mass_rownnz=model.M_rownnz,
      mass_colind=model.M_colind)
  assert layout["ncomponent"] == layout["ntree"] == 2
  assert layout["nnz"] == 2
