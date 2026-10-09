# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native distance queries over complete compiled collision operands.

Each source-ordered body/geom pair owns its CCD workspace. A separate reduction
publishes sensor outputs, so query ordering cannot alias scratch or omit assets
which physical collision masks exclude.
"""
from pathlib import Path
from functools import lru_cache
import mujoco
import numpy as np


def lower_geometry_queries(model, batch_size=1, limits=None):
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  from mujoco_metal.sensors import _sensor_geoms
  from mujoco_metal.coupled_constraints import lower_collision_geometry
  from mujoco_metal.common_ccd_bridge import estimate_production_upload_bytes, uses_pinned_mjc_convex
  if isinstance(batch_size,bool) or not isinstance(batch_size,(int,np.integer)) or batch_size<1:
    raise ValueError('geometry-query batch size must be a positive integer')
  kinds={int(getattr(mujoco.mjtSensor,'mjSENS_'+name))
         for name in ('GEOMDIST','GEOMNORMAL','GEOMFROMTO')}
  pairs=[]; bounds=[]; spans=np.zeros((model.nsensor,2),np.int32)
  for sensor in range(model.nsensor):
    if int(model.sensor_type[sensor]) not in kinds: continue
    start=len(pairs)
    for a in _sensor_geoms(model,model.sensor_objtype[sensor],model.sensor_objid[sensor]):
      for b in _sensor_geoms(model,model.sensor_reftype[sensor],model.sensor_refid[sensor]):
        if a==b: continue
        pairs.append((sensor,a,b)); bounds.append(float(model.sensor_cutoff[sensor]))
    spans[sensor]=start,len(pairs)-start
  count=len(pairs)
  if batch_size*count*10 >= (1<<31):
    raise CapacityOverflow('geometry-query result storage exceeds int32 addressing')
  scene=lower_collision_geometry(model,[(a,b) for _,a,b in pairs],limits=limits)
  common_bytes=estimate_production_upload_bytes(model,scene.mesh_hull,
                                               scene.mesh_hull_info,batch_size,count)
  terrain = any(int(model.geom_type[a])==1 or int(model.geom_type[b])==1
                for _,a,b in pairs)
  callbacks = terrain or (bool(int(model.opt.disableflags)&131072) and
      any(uses_pinned_mjc_convex(int(model.geom_type[a]),int(model.geom_type[b]))
          for _,a,b in pairs))
  terrain_bytes = (4*(12+55*count+2*batch_size+batch_size*count+
                        4*model.ngeom+27*batch_size*model.ngeom) if callbacks else 0)
  bytes_total=(terrain_bytes+common_bytes+batch_size*count*10*4+count*4*4+
               spans.nbytes+model.ngeom*4+6*4+batch_size*max(model.ngeom,1)*9*4)
  budget=CapacityLimits().memory_budget_bytes if limits is None else limits.memory_budget_bytes
  if bytes_total>budget:
    raise CapacityOverflow(f'geometry-query workspace {bytes_total} bytes exceeds budget {budget}')
  return {'scene':scene,'pairs':np.asarray(pairs,np.int32).reshape(-1,3),
          'cutoff':np.asarray(bounds,np.float32),'spans':spans,
          'common_bytes':common_bytes,'bytes_total':bytes_total,'terrain':terrain,'callbacks':callbacks}


@lru_cache(maxsize=2)
def _library(libccd):
  import torch
  root=Path(__file__).parent/'shaders'
  source=f'#define COMMON_CCD_USE_LIBCCD {int(libccd)}\n'
  for name in ('collision_primitives','common_ccd_core','common_native_ccd_support',
               'convex_narrowphase','plugin_sdf','sdf_narrowphase','libccd_mpr',
               'common_ccd_production','geometry_queries'):
    source+='\n'+(root/(name+'.metal')).read_text()
  return torch.mps.compile_shader(source)


class GeometryQueryProgram:
  def __init__(self,model,batch_size,limits=None):
    # All addressing/memory admission precedes device allocation.
    desc=lower_geometry_queries(model,batch_size,limits)
    from mujoco_metal.common_ccd_bridge import append_production_workspace
    scene=desc['scene']; count=len(desc['pairs'])
    hull,info,_=append_production_workspace(model,scene.mesh_hull,scene.mesh_hull_info,
                                          batch_size,count)
    if hull.nbytes+info.nbytes != desc['common_bytes']:
      raise RuntimeError('geometry-query preflight and actual allocation disagree')
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
      raise RuntimeError('geometry queries require native MPS compile_shader')
    self._completion=torch.mps.Event()
    self._torch=torch; self.model=model; self.batch_size=batch_size; self.count=count
    tensor=lambda a:torch.tensor(np.asarray(a),device='mps')
    self.hull=tensor(hull);self.info=tensor(info);self.pairs=tensor(desc['pairs'])
    self.cutoff=tensor(desc['cutoff']);self.spans=tensor(desc['spans'])
    self.types=tensor(np.asarray(model.geom_type,np.int32))
    self.dims=tensor(np.asarray([batch_size,count,model.ngeom,model.nsensor,
                                model.nsensordata,model.opt.disableflags],np.int32))
    self.results=torch.zeros((batch_size,count,10),dtype=torch.float32,device='mps')
    self.zeros=torch.zeros((batch_size,max(model.ngeom,1),9),dtype=torch.float32,device='mps')
    self._terrain=desc['terrain']
    self._callbacks=desc['callbacks']
    if self._callbacks:
      from mujoco_metal.coupled_constraints import _pack_contact_dims_offsets
      flags=int(model.opt.disableflags)
      packed=_pack_contact_dims_offsets([model.nv,count,50*count,batch_size,0,
          int(model.opt.cone),model.ngeom,0,int(bool(flags&(1<<19))),0],
          np.arange(count+1,dtype=np.int32)*50,np.zeros(50*count,np.int32))
      # Query selection is carried by logical_map; the common kernel's world
      # mask remains all enabled. It never determines numerical results on CPU.
      ptr=10+count+1+50*count
      packed=np.concatenate([packed[:ptr],np.asarray([ptr+1],np.int32),
                             np.ones(batch_size,np.int32)])
      self._terrain_dims=tensor(packed)
      canonical=desc['pairs'][:,1:].copy()
      for row in canonical:
        if int(model.geom_type[row[0]])>int(model.geom_type[row[1]]):
          row[0],row[1]=row[1],row[0]
      self._terrain_geoms=tensor(canonical.reshape(-1))
      self._terrain_margins=tensor(np.stack([desc['cutoff'],np.zeros(count,np.float32)],1).reshape(-1))
      self._terrain_sizes=tensor(np.asarray(model.geom_size,np.float32).reshape(-1))
      self._terrain_bounds=tensor(np.asarray(model.geom_rbound,np.float32).reshape(-1))
      self._terrain_map=torch.zeros(batch_size*count,dtype=torch.int32,device='mps')
      self._terrain_status=torch.zeros(batch_size,dtype=torch.int32,device='mps')
      self._terrain_matrices=torch.zeros((3,batch_size,model.ngeom,9),dtype=torch.float32,device='mps')
    self._libccd=bool(int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))

  def run(self,poses,metadata,stage_mask,out):
    if not self.count: return out
    t=self._torch; b=self.batch_size; ng=self.model.ngeom
    args=[]
    for name,width in (('geom_pos',3),('geom_pos_low',3),('geom_pos_tail',3),
                       ('geom_quat',4),('geom_xmat',9),('geom_xmat_low',9),
                       ('geom_xmat_tail',9)):
      value=poses.get(name)
      if value is None:
        if name in ('geom_pos','geom_quat'): raise ValueError(f'missing {name}')
        value=self.zeros.reshape(b,max(ng,1)*9)[:,:ng*width].contiguous()
      if (not isinstance(value,t.Tensor) or value.device.type!='mps'
          or value.dtype!=t.float32 or value.numel()!=b*ng*width
          or not value.is_contiguous()):
        raise ValueError(f'{name} must be contiguous float32 MPS geometry poses')
      args.append(value.reshape(-1))
    # When no source xmat is supplied, form its retained-precision quaternion
    # rotation in the shader; a zero matrix is only an explicit sentinel.
    library=_library(self._libccd)
    library.geometry_query_pairs(*args,self.types,self.pairs.reshape(-1),self.cutoff,
        self.hull,self.info,self.results.reshape(-1),stage_mask,self.dims,
        threads=(b*self.count,),group_size=(1,))
    if self._callbacks:
      self._terrain_status.zero_()
      library.geometry_query_prepare_terrain(args[3],args[4],args[5],args[6],
          self._terrain_matrices.reshape(-1),self._terrain_map,self.pairs.reshape(-1),
          stage_mask,self.dims,threads=(max(b*ng,b*self.count),),group_size=(1,))
      candidate_args=(args[0],args[3],args[1],args[2],
          *[self._terrain_matrices[i].reshape(-1) for i in range(3)],
          self._terrain_sizes,self.types,self._terrain_bounds,self._terrain_geoms,
          self._terrain_margins,self._terrain_dims,self._terrain_map,self.hull,self.info,
          self._terrain_status)
      if self._terrain:
        library.common_ccd_hfield_mesh_candidates(*candidate_args,
            threads=(b*self.count,),group_size=(1,))
      if self._libccd:
        for phase in range(5):
          library.common_ccd_rigid_convex_candidates(
              *candidate_args[:9],*candidate_args[10:],phase,
              threads=(b*self.count,),group_size=(1,))
      library.geometry_query_finish_terrain(self.types,self.pairs.reshape(-1),self.cutoff,
          self.hull,self.info,self.results.reshape(-1),stage_mask,self.dims,
          threads=(b*self.count,),group_size=(1,))
    library.geometry_query_reduce(self.results.reshape(-1),self.spans.reshape(-1),
        metadata,out.reshape(-1),stage_mask,self.dims,
        threads=(b*max(self.model.nsensor,1),),group_size=(1,))
    self._completion.record()
    self._completion.wait()
    return out
