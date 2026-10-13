# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source-rounded COM, CRBA and RNE operators on native Metal.

The compact ten-component inertia and operation grouping follow MuJoCo 3.10.
Only immutable compilation constants are processed on the host. Runtime poses,
velocities, masks, mass and bias remain on the device.
"""
from pathlib import Path
import numpy as np
from mujoco_metal.model import require_triple_word_domain
from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity


def source_rigid_position_words(model):
  return 3 * (23 * int(model.nbody) + 6 * int(model.nv))


def _source_constants(model, armature):
  nb, nv = int(model.nbody), int(model.nv)
  meta = np.concatenate([
      np.stack([model.body_parentid, model.body_rootid, model.body_jntadr,
                model.body_jntnum, model.body_dofadr, model.body_dofnum], axis=1).ravel(),
      np.stack([model.dof_parentid, model.dof_bodyid, model.dof_simplenum], axis=1).ravel(),
      np.stack([model.jnt_type, model.jnt_dofadr], axis=1).ravel()]).astype(np.int32)
  raw = np.concatenate([np.asarray(model.body_mass), np.asarray(model.body_subtreemass),
      np.asarray(model.body_inertia).ravel(), np.asarray(armature),
      np.asarray(model.dof_M0), np.asarray(model.gravity)]).astype(np.float64)
  require_triple_word_domain(raw, what='source rigid constants')
  hi=raw.astype(np.float32); lo=(raw-hi).astype(np.float32)
  tail=((raw-hi)-lo).astype(np.float32)
  return meta, np.stack([hi,lo,tail],axis=1).ravel()


class SourceRigidDynamics:
  def __init__(self, model, torch, device, armature):
    self.model,self.torch,self.device=model,torch,device
    meta,constants=_source_constants(model,armature)
    self.meta=torch.as_tensor(meta if meta.size else np.zeros(1,np.int32),device=device)
    self.constants=torch.as_tensor(constants,device=device)
    p=Path(__file__).parent/'shaders'
    source=('#define MUJOCO_METAL_SOURCE_MATH_ONLY\n'+(p/'common_ccd_core.metal').read_text()
       +'\n#undef MUJOCO_METAL_SOURCE_MATH_ONLY\n'+(p/'position_math.metal').read_text()
       +(p/'source_rigid.metal').read_text())
    self.library=torch.mps.compile_shader(source)

  def fork_workspace(self, batch):
    """Share immutable constants/shaders but own explicit-query scratch."""
    other=object.__new__(type(self))
    for name in ('model','torch','device','meta','constants','library'):
      setattr(other,name,getattr(self,name))
    other.prepare(batch)
    return other

  def prepare(self,batch):
    nb,nv=int(self.model.nbody),int(self.model.nv)
    pw=source_rigid_position_words(self.model);vw=3*(18*nb+6*nv)
    _validate_workspace_index_capacity(batch,{'source_rigid.position':batch*pw,
        'source_rigid.velocity':batch*vw},{'nbody':nb,'nv':nv})
    self.position_packet=self.torch.zeros((batch,pw),dtype=self.torch.float32,device=self.device)
    self.velocity_packet=self.torch.zeros((batch,vw),dtype=self.torch.float32,device=self.device)
    self.dims=self.torch.tensor([nb,nv,int(self.model.njnt),batch,
        pw//3,vw//3,int(self.model.disableflags)],dtype=self.torch.int32,device=self.device)

  def position(self,poses,workspace,lists,tree):
    w=workspace
    args=[self.meta,self.constants,self.position_packet,
      poses['body_quat'],poses['body_quat_low'],poses['body_quat_tail'],
      poses['inertial_pos'],poses['inertial_pos_low'],poses['inertial_pos_tail'],
      poses['inertial_quat'],poses['inertial_quat_low'],poses['inertial_quat_tail'],
      poses['joint_anchor'],poses['joint_anchor_low'],poses['joint_anchor_tail'],
      poses['joint_axis'],poses['joint_axis_low'],poses['joint_axis_tail'],
      w['mass'] if w['mass_storage']=='dense' else w['mass_blocks'],w['mass_low'],
      lists['body_ids'],lists['parent_ids'],lists['dof_ids'],lists['counts'],
      lists['body_mask'],lists['list_dims'],tree,w['mass_layout_dims'],self.dims]
    self.library.source_rigid_position(*args,threads=(w['batch_size'],),group_size=(1,))
    self.library.source_rigid_publish(self.meta,self.position_packet,w['root_com'],
      w['cdof'],w['local_inertia'],lists['body_ids'],lists['dof_ids'],
      lists['counts'],lists['list_dims'],w['mass_layout_dims'],self.dims,
      threads=(w['batch_size'],),group_size=(1,))

  def velocity(self,qvel_words,workspace,lists,world_mask,position_packet=None):
    w=workspace
    self.library.source_rigid_velocity(self.meta,self.constants,
      self.position_packet if position_packet is None else position_packet,
      self.velocity_packet,qvel_words,lists['body_ids'],lists['parent_ids'],
      lists['dof_ids'],lists['counts'],lists['list_dims'],world_mask,self.dims,
      w['bias'],w['bias_low'],w['cvel'],w['cvel_low'],w['cdof_dot'],
      w['cdof_dot_low'],w['cacc'],w['cacc_low'],w['body_force'],w['body_force_low'],
      threads=(w['batch_size'],),group_size=(1,))
