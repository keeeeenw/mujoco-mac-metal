# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Host admission of separately owned sensor buffers, without importing Torch."""
import mujoco
import numpy as np


def sensor_owned_workspace_sizes(model, batch_size, limits=None):
  """Exact constructor backings; aliases and borrowed simulation inputs count once.

  Returned counts are four-byte elements. Dispatch-local tensors are separately
  inventoried by the simulation's conservative peak-memory envelope.
  """
  from mujoco_metal.capacity import _integer_count, _check_i32_elements
  b = _integer_count('batch_size', batch_size, minimum=1)
  nq,nv,nj,nb,ng,ns,nt,nu = (int(getattr(model,n)) for n in
      ('nq','nv','njnt','nbody','ngeom','nsite','ntendon','nu'))
  nsen,ndata = int(model.nsensor),int(model.nsensordata)
  metadata = ('sensor_type','sensor_datatype','sensor_needstage','sensor_objtype',
      'sensor_objid','sensor_reftype','sensor_refid','sensor_dim','sensor_adr',
      'sensor_cutoff','jnt_type','jnt_qposadr','jnt_dofadr','body_iquat',
      'body_rootid','geom_bodyid','geom_pos','geom_quat','site_bodyid',
      'site_pos','site_quat')
  sizes = {'meta.'+name:int(np.asarray(getattr(model,name)).size)
           for name in metadata}
  sizes.update(dims=10,stage_mask=1+b*max(nsen,1)+b,output=b*ndata,
      sensor_meta=nsen*10,jnt_meta=max(nj,1)*4,jnt_lim=max(nj,1)*3,
      ten_lim=max(nt,1)*9,massub=max(nb,1)*5,body_tree=max(nb,1)*2,
      site_geom=max(ns,1)*4,econst=11+nq+3*nj+34*int(model.ncam),
      ten_fix_lmap=max(nt,1)*max(nq,1),ten_fix_mmap=max(nt,1)*max(nv,1),
      act_stat_lmap=max(nu,1)*max(nq,1),act_stat_mmap=max(nu,1)*max(nv,1),
      subtree_runtime=4+b*nb*32+b*max(nsen,1),dummy=max(b*nt,b*nu,1),
      state_output_pair=2*max(b*ndata,1),rne_body_jnt=max(nb,1)*4,
      rne_dof_jntid=max(nv,1),rne_jnt_type=max(nj,1),rne_jnt_dofadr=max(nj,1),
      rne_mass=max(nb,1),rne_body_tree=max(nb,1)*2,rne_inertia=max(nb,1)*3,
      rne_gravity=3,rne_body_weld=max(nb,1),rne_geom_bodyid=max(ng,1),
      rne_site_bodyid=max(ns,1),rne_eq_meta=max(int(model.neq),1)*4,
      rne_eq_data=max(int(model.neq),1)*11,rne_site_lpos=max(ns,1)*3,
      rne_act_trn=max(nu,1)*2,rne_cacc_pair=2*b*max(nb,1)*6,
      rne_cfrc=b*max(nb,1)*6,rne_scom=b*max(nb,1)*3,
      rne_ext=b*max(nb,1)*6,rne_com_scratch=b*max(nb,1)*12,
      rne_qacc_low_zero=b*max(nv,1))
  spatial_names=('TOUCH','CONTACT','TACTILE','RANGEFINDER','GEOMDIST',
                 'GEOMNORMAL','GEOMFROMTO')
  spatial = np.isin(model.sensor_type,[int(getattr(mujoco.mjtSensor,
      'mjSENS_'+name)) for name in spatial_names]).any()
  if spatial:
    sizes.update(sp_geom_type=max(ng,1),sp_geom_size=max(ng,1)*3,
        sp_geom_bodyid=max(ng,1),sp_geom_matid=max(ng,1),sp_geom_rgba=max(ng,1)*4,
        sp_mat_rgba=max(int(model.nmat),1)*4,sp_body_geoms=max(nb,1)*2,
        sp_geom_rbound=max(ng,1))
    from mujoco_metal.sensors import tactile_workspace_sizes
    sizes.update({'tactile.'+n:c for n,c in tactile_workspace_sizes(model).items()})
  distance = np.isin(model.sensor_type,[int(getattr(mujoco.mjtSensor,
      'mjSENS_'+n)) for n in ('GEOMDIST','GEOMNORMAL','GEOMFROMTO')]).any()
  if distance:
    from mujoco_metal.geometry_queries import lower_geometry_queries
    sizes.update({'geometry_query.'+n:c for n,c in
        lower_geometry_queries(model,b,limits)['workspace_sizes'].items()})
  # The initial nine-word info backing is replaced by the query's owned info.
  # Its construction peak is accounted separately by the simulation envelope.
  else:
    sizes['sp_hull_info']=9
  from mujoco_metal.ray_queries import site_ray_workspace_sizes
  sizes.update({'site_rays.'+n:c for n,c in site_ray_workspace_sizes(model).items()})
  for name,count in sizes.items():
    _check_i32_elements('sensor.'+name,count)
  return sizes


def preflight_sensor_workspace(model, batch_size, limits=None):
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  sizes=sensor_owned_workspace_sizes(model,batch_size,limits)
  budget=(CapacityLimits() if limits is None else limits).memory_budget_bytes
  requested=4*sum(sizes.values())
  if requested>budget:
    raise CapacityOverflow(f'sensor workspace {requested} bytes exceeds budget {budget}')
  return sizes
