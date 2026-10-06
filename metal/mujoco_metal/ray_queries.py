# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native scene rays against pinned MuJoCo 3.10 geometry.

Host model compilation supplies immutable topology/assets. Query poses, ray
intersection, filtering and nearest-hit reduction execute on Metal. Assets are
packed dynamically, including non-convex meshes and non-colliding geoms; contact
candidate capacities do not restrict this scene-query path.
"""
from dataclasses import dataclass
from pathlib import Path
import numbers

import mujoco
import numpy as np


def _frozen(value, dtype):
  value = np.asarray(value, dtype=dtype)
  return np.frombuffer(value.tobytes(), dtype=dtype).reshape(value.shape)


@dataclass(frozen=True)
class RayScene:
  ngeom: int
  metadata: np.ndarray
  sizes: np.ndarray
  hull_info: np.ndarray
  hull: np.ndarray
  octree_info: np.ndarray
  octree: np.ndarray
  sdf_kind: np.ndarray
  sdf_attributes: np.ndarray
  sdf_attributes_low: np.ndarray
  rbound: np.ndarray

  @property
  def nbytes(self):
    return sum(value.nbytes for value in vars(self).values()
               if isinstance(value, np.ndarray))


def lower_ray_scene(model):
  """Lower scene assets without importing Torch or executing engine callbacks."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("ray queries require pinned MuJoCo 3.10.0")
  ng = int(model.ngeom)
  metadata = np.zeros((max(ng, 1), 6), dtype=np.int32)
  info = np.full((max(ng, 1), 9), -1, dtype=np.int32)
  octinfo = np.full((max(ng, 1), 2), -1, dtype=np.int32)
  kind = np.zeros(max(ng, 1), dtype=np.int32)
  attrs = np.zeros((max(ng, 1), 5), dtype=np.float32)
  attrs_low = np.zeros_like(attrs)
  hull, octree = [], []
  for g in range(ng):
    body, mat = int(model.geom_bodyid[g]), int(model.geom_matid[g])
    alpha = model.geom_rgba[g, 3] if mat < 0 else model.mat_rgba[mat, 3]
    metadata[g] = [int(model.geom_type[g]), body,
                   int(model.body_weldid[body] == 0),
                   min(5, max(0, int(model.geom_group[g]))), int(alpha != 0),
                   int(model.body_bvhadr[body] >= 0)]
  # Mesh triangle queries use the original compiled face set, without convex
  # contact hull eligibility or fixed contact vertex/face limits.
  for g in range(ng):
    if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
      continue
    mesh = int(model.geom_dataid[g])
    va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    vertices = np.asarray(model.mesh_vert[va:va+vn], np.float32).reshape(-1)
    faces = np.asarray(model.mesh_face[fa:fa+fn], np.int64).reshape(-1)
    if np.any(faces < 0) or np.any(faces >= vn) or vn >= 2**24:
      raise ValueError("mesh face indices must be valid and float32-exact")
    info[g, 0:2] = [len(hull)//3, vn]
    hull.extend(vertices.tolist())
    info[g, 3:5] = [len(hull)//3, fn]
    hull.extend(faces.astype(np.float32).tolist())
  for g in range(ng):
    if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_HFIELD):
      field = int(model.geom_dataid[g])
      rows, cols = int(model.hfield_nrow[field]), int(model.hfield_ncol[field])
      info[g, 8] = len(hull)
      hull.extend(np.asarray(model.hfield_size[field], np.float32).tolist())
      info[g, 5:8] = [len(hull), rows, cols]
      adr = int(model.hfield_adr[field])
      hull.extend(np.asarray(model.hfield_data[adr:adr+rows*cols], np.float32).tolist())
    if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SDF):
      mesh = int(model.geom_dataid[g])
      if int(model.geom_plugin[g]) >= 0:
        from mujoco_metal.bundled_plugins import lower_bundled_plugins
        descriptor = lower_bundled_plugins(model)
        instance = int(model.geom_plugin[g])
        kind[g] = descriptor.instance_kind[instance]
        if not 1 <= int(kind[g]) <= 5:
          raise ValueError("geom plugin has no native SDF evaluator")
        attrs[g] = descriptor.plugin_attributes[instance]
        attrs_low[g] = descriptor.plugin_attributes_low[instance]
      else:
        adr, count = int(model.mesh_octadr[mesh]), int(model.mesh_octnum[mesh])
        if adr < 0 or count <= 0 or count >= 2**24:
          raise ValueError("SDF geom needs a compiled octree or native plugin")
        octinfo[g] = [len(octree), count]
        octree.extend(np.concatenate([
            model.oct_child[adr:adr+count].reshape(-1),
            model.oct_aabb[adr:adr+count].reshape(-1),
            model.oct_coeff[adr:adr+count].reshape(-1)]).astype(np.float32).tolist())
  fields = {
      'metadata': metadata, 'sizes': np.asarray(model.geom_size) if ng else np.zeros((1, 3)),
      'hull_info': info, 'hull': hull or [0.0], 'octree_info': octinfo,
      'octree': octree or [0.0], 'sdf_kind': kind, 'sdf_attributes': attrs,
      'sdf_attributes_low': attrs_low,
      'rbound': np.asarray(model.geom_rbound) if ng else np.zeros(1),
  }
  for name, value in fields.items():
    dtype = np.int32 if name in ('metadata', 'hull_info', 'octree_info', 'sdf_kind') else np.float32
    fields[name] = _frozen(value, dtype)
    if not np.all(np.isfinite(fields[name])):
      raise ValueError(f"nonfinite compiled ray asset: {name}")
    if fields[name].size > 2**31-1:
      raise ValueError(f"ray asset exceeds signed int32 indexing: {name}")
  return RayScene(ng, **fields)


def _compile_ray_library(torch):
  directory = Path(__file__).parent / 'shaders'
  from mujoco_metal.coupled_constraints import _COLLISION_SHADER, _CONVEX_SHADER, _SDF_SHADER
  paths = (_COLLISION_SHADER, _CONVEX_SHADER, _SDF_SHADER,
           directory/'sensors_spatial.metal', directory/'plugin_sdf.metal', directory/'ray_queries.metal')
  return torch.mps.compile_shader('\n'.join(path.read_text() for path in paths))


class MetalRayQueries:
  """Owned batched hit distance, geom ID, normal and per-ray status outputs.

  Status 1 means a nonfinite ray/pose or a direction shorter than mjMINVAL;
  status 2 means invalid SDF arithmetic. No hit has distance/geom ID -1 and
  zero normal with status 0. Ray directions retain upstream non-unit semantics.
  """

  def __init__(self, model, batch_size, memory_budget_bytes=1 << 30):
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    if (isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, numbers.Integral)
        or memory_budget_bytes <= 0):
      raise ValueError("memory_budget_bytes must be a positive integer")
    self.scene = lower_ray_scene(model)
    self.model, self.batch_size = model, int(batch_size)
    self.memory_budget_bytes = int(memory_budget_bytes)
    if self.scene.nbytes > self.memory_budget_bytes:
      raise ValueError("ray scene exceeds memory budget before device allocation")
    if self.batch_size * max(self.scene.ngeom, 1)*4 > 2**31-1:
      raise ValueError("ray pose storage exceeds signed int32 indexing")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, 'compile_shader'):
      raise RuntimeError("ray queries require PyTorch MPS compile_shader")
    self._torch, self.device = torch, torch.device('mps')
    self._constants = {name: torch.as_tensor(np.array(value, copy=True), device=self.device)
                       for name, value in vars(self.scene).items() if isinstance(value, np.ndarray)}
    # Shared intersection/SDF helpers; no separate per-demo physics path.
    self._library = _compile_ray_library(torch)
    self._kernel = self._library.query_scene_rays

  def _tensor(self, value, shape, name):
    torch = self._torch
    if (not isinstance(value, torch.Tensor) or value.device.type != 'mps'
        or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape)
        or not value.is_contiguous()):
      raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    return value

  def run_device(self, poses, origins, vectors, *, geomgroup=None,
                 flg_static=True, bodyexclude=-1, geomid=-1, cutoff=None,
                 multiray=False):
    torch, b, ng = self._torch, self.batch_size, self.scene.ngeom
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 3:
      raise ValueError("vectors must have shape [batch,nray,3]")
    n = int(vectors.shape[1])
    if b*n*3 > 2**31-1:
      raise ValueError("ray outputs exceed signed int32 indexing")
    # Four outputs (six 32-bit words per ray) and all transient bindings:
    # dims[8], geomgroup[6], cutoff[1], and the empty-scene pose sentinel.
    parameter_bytes = (8 + 6 + 1 + int(ng == 0)) * 4
    if self.scene.nbytes + b*n*24 + parameter_bytes > self.memory_budget_bytes:
      raise ValueError("ray outputs exceed memory budget before device allocation")
    self._tensor(vectors, (b, n, 3), 'vectors')
    self._tensor(origins, (b, n, 3), 'origins')
    for name, width in (('geom_pos', 3), ('geom_quat', 4)):
      self._tensor(poses[name], (b, ng, width), name)
    for value, count, name in ((bodyexclude, self.model.nbody, 'bodyexclude'),
                               (geomid, ng, 'geomid')):
      if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not -1 <= value < count:
        raise ValueError(f"{name} must be -1 or a compiled object ID")
    if not isinstance(flg_static, (bool, np.bool_)):
      raise TypeError("flg_static must be boolean")
    if not isinstance(multiray, (bool, np.bool_)):
      raise TypeError("multiray must be boolean")
    groups = np.ones(6, dtype=np.int32) if geomgroup is None else np.asarray(geomgroup)
    if groups.shape != (6,) or not np.issubdtype(groups.dtype, np.integer):
      raise ValueError("geomgroup must have six integer entries")
    groups = (groups != 0).astype(np.int32)
    limit = 0.0 if cutoff is None else float(cutoff)
    if cutoff is not None and (not np.isfinite(limit) or limit < 0 or not np.isfinite(np.float32(limit))):
      raise ValueError("cutoff must be finite, nonnegative and float32-representable")
    distance = torch.empty((b, n), dtype=torch.float32, device=self.device)
    hit = torch.empty((b, n), dtype=torch.int32, device=self.device)
    normal = torch.empty((b, n, 3), dtype=torch.float32, device=self.device)
    status = torch.empty((b, n), dtype=torch.int32, device=self.device)
    if n == 0:
      return dict(distance=distance, geomid=hit, normal=normal, status=status)
    dims = torch.tensor([b, n, ng, int(flg_static), int(bodyexclude), int(geomid),
                         int(cutoff is not None), int(multiray)], dtype=torch.int32, device=self.device)
    filters = torch.as_tensor(groups, device=self.device)
    bound = torch.tensor([limit], dtype=torch.float32, device=self.device)
    c = self._constants
    dummy = torch.zeros(1, device=self.device) if ng == 0 else None
    self._kernel(poses['geom_pos'] if ng else dummy, poses['geom_quat'] if ng else dummy,
                 c['metadata'], c['sizes'], c['hull'], c['hull_info'],
                 c['octree'], c['octree_info'], c['sdf_kind'],
                 c['sdf_attributes'], c['sdf_attributes_low'], c['rbound'],
                 origins, vectors, filters, bound, dims,
                 distance, hit, normal, status, threads=(b*n,), group_size=(1,))
    return dict(distance=distance, geomid=hit, normal=normal, status=status)


class MetalRayPrimitives:
  """Batched pinned ``mju_rayGeom`` math with explicit, caller-supplied poses.

  This utility uses no compiled scene assets or simulation workspaces. Matrices
  have row-major shape [B,3,3]. Results are owned MPS tensors; invalid finite
  input/arithmetic is isolated by world through status 1/2 respectively.
  """

  def __init__(self, batch_size, memory_budget_bytes=1 << 30):
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
      raise ValueError('batch_size must be a positive integer')
    if (isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, numbers.Integral)
        or memory_budget_bytes <= 0):
      raise ValueError('memory_budget_bytes must be a positive integer')
    self.batch_size = int(batch_size)
    if self.batch_size*9 > 2**31-1:
      raise ValueError('primitive ray storage exceeds signed int32 indexing')
    # Outputs: distance, normal[3], status. Parameters: batch, geom type.
    if self.batch_size*20+8 > int(memory_budget_bytes):
      raise ValueError('primitive ray outputs exceed memory budget before device allocation')
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, 'compile_shader'):
      raise RuntimeError('ray queries require PyTorch MPS compile_shader')
    self._torch, self.device = torch, torch.device('mps')
    self._library = _compile_ray_library(torch)
    self._kernel = self._library.query_analytic_rays

  def run_device(self, pos, mat, size, origins, vectors, geomtype):
    if (isinstance(geomtype, (bool, np.bool_)) or not isinstance(geomtype, numbers.Integral)
        or int(geomtype) not in (0, 2, 3, 4, 5, 6)):
      raise ValueError('geomtype must be plane, sphere, capsule, ellipsoid, cylinder or box')
    b, torch = self.batch_size, self._torch
    for value, shape, name in ((pos,(b,3),'pos'), (mat,(b,3,3),'mat'),
                              (size,(b,3),'size'), (origins,(b,3),'origins'),
                              (vectors,(b,3),'vectors')):
      MetalRayQueries._tensor(self, value, shape, name)
    distance = torch.empty((b,), dtype=torch.float32, device=self.device)
    normal = torch.empty((b,3), dtype=torch.float32, device=self.device)
    status = torch.empty((b,), dtype=torch.int32, device=self.device)
    dims = torch.tensor([b, int(geomtype)], dtype=torch.int32, device=self.device)
    self._kernel(pos,mat,size,origins,vectors,dims,distance,normal,status,
                 threads=(b,),group_size=(1,))
    return dict(distance=distance,normal=normal,status=status)


@dataclass(frozen=True)
class RaySurface:
  """Immutable local vertex topology; intersections never run on the host."""
  nvert: int
  dim: int
  radius: float
  edges: np.ndarray
  faces: np.ndarray
  layers: np.ndarray

  @property
  def nbytes(self):
    return self.edges.nbytes+self.faces.nbytes+self.layers.nbytes


def lower_ray_skin(nvert, faces):
  if isinstance(nvert, bool) or not isinstance(nvert, numbers.Integral) or nvert < 0:
    raise ValueError('nvert must be a nonnegative integer')
  faces = np.asarray(faces)
  if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
    raise ValueError('faces must be integer[nface,3]')
  if nvert*3 > 2**31-1 or faces.size > 2**31-1:
    raise ValueError('surface topology exceeds signed int32 indexing')
  if np.any(faces < 0) or np.any(faces >= nvert):
    raise ValueError('faces must reference local surface vertices')
  return RaySurface(int(nvert),2,0.0,_frozen(np.empty((0,2)),np.int32),
      _frozen(faces,np.int32),_frozen(np.zeros(len(faces)),np.int32))


def lower_ray_flex(model, flexid):
  if not isinstance(model, mujoco.MjModel):
    raise TypeError('model must be a compiled mujoco.MjModel')
  if mujoco.__version__ != '3.10.0':
    raise RuntimeError('flex ray queries require pinned MuJoCo 3.10.0')
  if (isinstance(flexid,(bool,np.bool_)) or not isinstance(flexid,numbers.Integral)
      or not 0 <= flexid < int(model.nflex)):
    raise ValueError('flexid must identify a compiled flex')
  f = int(flexid)
  dim, nvert = int(model.flex_dim[f]), int(model.flex_vertnum[f])
  ea, en = int(model.flex_edgeadr[f]), int(model.flex_edgenum[f])
  edges = np.asarray(model.flex_edge[ea:ea+en])
  faces, layers = [], []
  if dim > 1:
    adr, count = int(model.flex_elemdataadr[f]), int(model.flex_elemnum[f])
    data = np.asarray(model.flex_elem[adr:adr+count*(dim+1)]).reshape(count,dim+1)
    la = int(model.flex_elemadr[f])
    mapping = ((0,1,2),) if dim == 2 else ((0,1,2),(0,1,3),(0,2,3),(1,2,3))
    for e, element in enumerate(data):
      for ids in mapping:
        faces.append(element[list(ids)])
        layers.append(int(model.flex_elemlayer[la+e]))
  faces = np.asarray(faces,np.int32).reshape(-1,3)
  radius = float(model.flex_radius[f])
  if not np.isfinite(radius) or radius < 0 or not np.isfinite(np.float32(radius)):
    raise ValueError('flex radius must be finite, nonnegative and float32 representable')
  if (nvert*3 > 2**31-1 or edges.size > 2**31-1 or faces.size > 2**31-1
      or np.any(edges < 0) or np.any(edges >= nvert)
      or np.any(faces < 0) or np.any(faces >= nvert)):
    raise ValueError('invalid or overflowing local flex ray topology')
  return RaySurface(nvert,dim,radius,_frozen(edges,np.int32),
      _frozen(faces,np.int32),_frozen(layers,np.int32))


class MetalRaySurface:
  """Pinned flex/skin ray selection over current device vertex positions.

  Local vertex IDs, source face order, strict hit ordering, capsule-endpoint
  ties, face ties, tetrahedron layers and skin flags follow engine_ray.c.
  No fixed vertex/face capacity or collision eligibility applies.
  """

  def __init__(self, surface, batch_size, memory_budget_bytes=1 << 30):
    if not isinstance(surface,RaySurface):
      raise TypeError('surface must be lowered immutable ray topology')
    if (isinstance(surface.nvert,(bool,np.bool_)) or not isinstance(surface.nvert,numbers.Integral)
        or surface.nvert < 0 or surface.dim not in (1,2,3)
        or not np.isfinite(surface.radius) or surface.radius < 0
        or not np.isfinite(np.float32(surface.radius))):
      raise ValueError('surface dimensions and radius must be valid')
    for a,width,name in ((surface.edges,2,'edges'),(surface.faces,3,'faces')):
      if (not isinstance(a,np.ndarray) or a.dtype != np.int32 or a.ndim != 2
          or a.shape[1] != width or a.size > 2**31-1
          or np.any(a<0) or np.any(a>=surface.nvert)):
        raise ValueError(f'{name} must be valid int32 local vertex topology')
    if (not isinstance(surface.layers,np.ndarray) or surface.layers.dtype != np.int32
        or surface.layers.shape != (len(surface.faces),)):
      raise ValueError('layers must match the int32 face topology')
    # Own immutable copies even when a caller constructs RaySurface directly.
    surface = RaySurface(int(surface.nvert),int(surface.dim),float(surface.radius),
        _frozen(surface.edges,np.int32),_frozen(surface.faces,np.int32),
        _frozen(surface.layers,np.int32))
    if isinstance(batch_size,bool) or not isinstance(batch_size,numbers.Integral) or batch_size <= 0:
      raise ValueError('batch_size must be a positive integer')
    if (isinstance(memory_budget_bytes,bool) or not isinstance(memory_budget_bytes,numbers.Integral)
        or memory_budget_bytes <= 0):
      raise ValueError('memory_budget_bytes must be a positive integer')
    b = int(batch_size)
    if b*max(surface.nvert,1)*3 > 2**31-1:
      raise ValueError('surface ray storage exceeds signed int32 indexing')
    # Owned outputs24B/world; dims10, radius1; empty buffers bind one word.
    required = surface.nbytes+b*24+44+sum(4 for a in (
        surface.edges,surface.faces,surface.layers) if a.size == 0)
    if required > int(memory_budget_bytes):
      raise ValueError('surface ray outputs exceed memory budget before device allocation')
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
      raise RuntimeError('ray queries require PyTorch MPS compile_shader')
    self.surface, self.batch_size = surface, b
    self._torch, self.device = torch, torch.device('mps')
    self._constants = {
        name:torch.as_tensor(np.array(a,copy=True) if a.size else np.zeros(1,np.int32),
                             device=self.device)
        for name,a in (('edges',surface.edges),('faces',surface.faces),('layers',surface.layers))}
    self._radius = torch.tensor([surface.radius],dtype=torch.float32,device=self.device)
    self._library = _compile_ray_library(torch)
    self._kernel = self._library.query_surface_rays

  def run_device(self, vertices, origins, vectors, *, flex_layer=0,
                 flg_vert=False, flg_edge=False, flg_face=True, flg_skin=False):
    b, s, torch = self.batch_size, self.surface, self._torch
    if (isinstance(flex_layer,(bool,np.bool_)) or not isinstance(flex_layer,numbers.Integral)
        or not -(2**31) <= flex_layer < 2**31):
      raise ValueError('flex_layer must be a signed int32 integer')
    for name,flag in (('flg_vert',flg_vert),('flg_edge',flg_edge),
                      ('flg_face',flg_face),('flg_skin',flg_skin)):
      if not isinstance(flag,(bool,np.bool_)):
        raise TypeError(f'{name} must be boolean')
    for value,shape,name in ((vertices,(b,s.nvert,3),'vertices'),
                            (origins,(b,3),'origins'),(vectors,(b,3),'vectors')):
      MetalRayQueries._tensor(self,value,shape,name)
    dims = torch.tensor([b,s.nvert,len(s.edges),len(s.faces),s.dim,
        int(flg_vert),int(flg_edge),int(flg_face),int(flg_skin),int(flex_layer)],
        dtype=torch.int32,device=self.device)
    distance = torch.empty(b,dtype=torch.float32,device=self.device)
    vertex = torch.empty(b,dtype=torch.int32,device=self.device)
    normal = torch.empty((b,3),dtype=torch.float32,device=self.device)
    status = torch.empty(b,dtype=torch.int32,device=self.device)
    # Empty vertex topology exits without reading its binding.
    binding = vertices if s.nvert else self._radius
    c = self._constants
    self._kernel(binding,c['edges'],c['faces'],c['layers'],self._radius,
                 origins,vectors,dims,distance,vertex,normal,status,
                 threads=(b,),group_size=(1,))
    return dict(distance=distance,vertid=vertex,normal=normal,status=status)
