# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native gate for opt-in independent-tree inertia blocks and matvec."""

import os

import mujoco
import numpy as np
import pytest


XML = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
  <body name="a" pos="-1 0 1"><freejoint name="a_free"/><geom type="box" size=".1 .2 .3" mass="2"/>
    <body pos="0 0 .4"><joint name="a_hinge" type="hinge" axis="0 1 0"/><geom type="sphere" size=".1" mass=".5"/></body>
  </body>
  <body name="b" pos="1 0 1"><freejoint/><geom type="sphere" size=".2" mass="1"/></body>
</worldbody></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_sparse_tree_blocks_and_device_matvec_match_cpu_mass_oracle():
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model = mujoco.MjModel.from_xml_string(XML)
  descriptor = load_model(model)
  batch = 2
  smooth = MetalSmoothDynamics(descriptor, batch_size=batch,
                               mass_storage="block_sparse")
  qpos_np = np.tile(np.asarray(model.qpos0, dtype=np.float64), (batch, 1))
  for world, angle in enumerate((0.37, -0.52)):
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "a_free")
    adr = int(model.jnt_qposadr[free])
    qpos_np[world, adr + 3:adr + 7] = (
        np.cos(angle / 2), 0, np.sin(angle / 2), 0)
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "a_hinge")
    qpos_np[world, int(model.jnt_qposadr[hinge])] = angle / 2
  qpos = torch.as_tensor(qpos_np.astype(np.float32), device="mps")
  qvel = torch.zeros((batch, model.nv), dtype=torch.float32, device="mps")
  result = smooth.run_device(qpos, qvel)
  assert "mass_matrix" not in result
  blocks = result["mass_blocks"]
  layout = result["mass_block_layout"]
  vector_np = np.linspace(-0.8, 1.1, batch * model.nv,
                          dtype=np.float32).reshape(batch, model.nv)
  vector = torch.as_tensor(vector_np, device="mps")
  product = smooth.mass_blocks_matvec_device(blocks, vector)
  product_host = product.cpu().numpy()
  blocks_host = blocks.cpu().numpy()

  for world in range(batch):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_np[world]
    mujoco.mj_forward(model, data)
    dense = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, dense)
    reconstructed = np.zeros_like(dense)
    for start, width, offset in zip(
        layout["tree_dofadr"], layout["tree_dofnum"],
        layout["tree_mass_offsets"]):
      start, width, offset = int(start), int(width), int(offset)
      if width:
        block = blocks_host[world, offset:offset + width * width]
        reconstructed[start:start + width, start:start + width] = block.reshape(width, width)
    np.testing.assert_allclose(reconstructed, dense, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(product_host[world], dense @ vector_np[world],
                               rtol=5e-5, atol=5e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("storage", ["dense", "block_sparse"])
def test_tendon_armature_retains_asleep_component_entries(storage):
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option><flag sleep="enable"/></option><worldbody>
      <body name="a" pos="-1 0 0" sleep="allowed">
        <joint name="ja" type="slide"/><geom type="sphere" size=".1" mass="1"/>
      </body>
      <body name="b" pos="1 0 0" sleep="allowed">
        <joint name="jb" type="slide"/><geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody><tendon><fixed name="t" armature="1" limited="false">
      <joint joint="ja" coef="1"/><joint joint="jb" coef="1"/>
    </fixed></tendon></mujoco>""")
  smooth = MetalSmoothDynamics(load_model(model), batch_size=1,
                               mass_storage=storage)
  qpos = torch.as_tensor(model.qpos0.astype(np.float32)[None, :], device="mps")
  qvel = torch.zeros((1, model.nv), dtype=torch.float32, device="mps")
  lists = dict(smooth._workspace["all_awake_lists"])
  lists["tree_awake"] = torch.ones(
      (1, max(smooth.ntree, 1)), dtype=torch.int32, device="mps")
  first_j = torch.as_tensor([[[1.0, 2.0]]], dtype=torch.float32, device="mps")
  first = smooth.run_device(qpos, qvel, awake_lists=lists,
                            tendon_J=first_j)
  name = ("tendon_armature_matrix" if storage == "dense"
          else "tendon_armature_blocks")
  first_entries = first[name].clone()
  assert bool(torch.any(first_entries != 0).item())

  # No tendon tree is awake on this call. The new J values must not rewrite
  # cached armature terms belonging to the sleeping component.
  lists["tree_awake"].zero_()
  changed_j = torch.as_tensor([[[3.0, 4.0]]], dtype=torch.float32, device="mps")
  asleep = smooth.run_device(qpos, qvel, awake_lists=lists,
                             tendon_J=changed_j)
  torch.testing.assert_close(asleep[name], first_entries, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_sparse_tendon_armature_matches_pinned_three_tree_sparse_addition():
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  template = """<mujoco><option gravity="0 0 0"/><worldbody>
    <body pos="-1 0 0"><joint name="a" type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
    <body><joint name="b" type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
    <body pos="1 0 0"><joint name="c" type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody><tendon><fixed name="t" armature="1" limited="false">
      <joint joint="a" coef="{a}"/><joint joint="b" coef="{b}"/>
      <joint joint="c" coef="{c}"/></fixed></tendon></mujoco>"""
  xml = template.format(a=1.0, b=-2.0, c=.5)
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = load_model(model)
  smooth = MetalSmoothDynamics(descriptor, batch_size=2,
                               mass_storage="block_sparse")
  qpos_np = np.tile(np.asarray(model.qpos0, dtype=np.float32), (2, 1))
  qvel = torch.zeros((2, model.nv), dtype=torch.float32, device="mps")
  jac = np.asarray([[1.0, -2.0, .5], [-.25, .75, 2.0]], dtype=np.float32)
  tendon_j = torch.as_tensor(jac[:, None, :], dtype=torch.float32,
                             device="mps").contiguous()
  result = smooth.run_device(
      torch.as_tensor(qpos_np, device="mps"), qvel, tendon_J=tendon_j)
  layout = result["mass_block_layout"]
  blocks = result["tendon_armature_blocks"].cpu().numpy()

  for world in range(2):
    # The producer input is explicitly world-specific. Build a matching CPU
    # oracle model for each row so the test checks the exact compiled M CSR
    # sparsity pattern and these supplied row values independently.
    oracle_model = mujoco.MjModel.from_xml_string(template.format(
        a=float(jac[world, 0]), b=float(jac[world, 1]),
        c=float(jac[world, 2])))
    oracle_data = mujoco.MjData(oracle_model)
    oracle_data.qpos[:] = qpos_np[world]
    mujoco.mj_forward(oracle_model, oracle_data)
    expected = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(oracle_model, oracle_data, expected)
    zero_armature_model = mujoco.MjModel.from_xml_string(
        template.format(a=float(jac[world, 0]), b=float(jac[world, 1]),
                        c=float(jac[world, 2])).replace(
                            'armature="1"', 'armature="0"'))
    zero_armature_data = mujoco.MjData(zero_armature_model)
    zero_armature_data.qpos[:] = qpos_np[world]
    mujoco.mj_forward(zero_armature_model, zero_armature_data)
    base_mass = np.zeros_like(expected)
    mujoco.mj_fullM(zero_armature_model, zero_armature_data, base_mass)
    expected -= base_mass

    reconstructed = np.zeros_like(expected)
    offsets = layout["component_dof_offsets"]
    dofs = layout["component_dof_ids"]
    mass_offsets = layout["component_mass_offsets"]
    for component in range(int(layout["ncomponent"])):
      begin, end = int(offsets[component]), int(offsets[component + 1])
      ids = dofs[begin:end]
      width = end - begin
      start = int(mass_offsets[component])
      block = blocks[world, start:start + width * width].reshape(width, width)
      reconstructed[np.ix_(ids, ids)] = block
    # Rebuilding from a matching CPU model ensures off-pattern outer-product
    # entries remain absent exactly as they are in pinned compressed M.
    np.testing.assert_allclose(reconstructed, expected, rtol=3e-5, atol=3e-5)
