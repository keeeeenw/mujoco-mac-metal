# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compile the collision hull already produced by pinned MuJoCo preprocessing.

The graph layout follows MuJoCo 3.10.0 user_mesh.cc:MakeGraph. Collision hulls
are distinct from the original triangle surface used by ray and SDF queries.
This host-only lowering performs no simulation-time physics or GPU readback.
"""

import numpy as np


def collision_mesh_hull(model, meshid):
  """Return owned float32 hull vertices and local int32 triangle indices.

Use the compiler's actual hull, including maxhullvert simplification. Original
concave faces must never be used as hull snap planes. If the compiler omitted
the graph, the original surface is accepted only after convexity validation.
"""
  if isinstance(meshid, bool) or not isinstance(meshid, (int, np.integer)):
    raise TypeError("mesh id must be an integer")
  meshid = int(meshid)
  if not 0 <= meshid < int(model.nmesh):
    raise ValueError("mesh id is outside the compiled model")
  vadr, vnum = int(model.mesh_vertadr[meshid]), int(model.mesh_vertnum[meshid])
  vertices = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float32)
  if vertices.shape != (vnum, 3) or vnum <= 0 or not np.isfinite(vertices).all():
    raise ValueError("compiled mesh vertices are malformed or nonfinite")
  graph = np.asarray(model.mesh_graph, dtype=np.int64).reshape(-1)
  adr = int(model.mesh_graphadr[meshid])
  if adr >= 0:
    # numvert,numface,edgeadr[nv],globalid[nv],edges[nv+3*nf],faces[3*nf].
    if adr + 2 > len(graph):
      raise ValueError("compiled collision hull header is truncated")
    nv, nf = map(int, graph[adr:adr + 2])
    end = adr + 2 + 3 * nv + 6 * nf
    next_adrs = np.asarray(model.mesh_graphadr, dtype=np.int64)
    following = next_adrs[next_adrs > adr]
    limit = min(len(graph), int(following.min())) if len(following) else len(graph)
    if nv <= 0 or nf <= 0 or end > limit:
      raise ValueError("compiled collision hull exceeds its graph storage")
    ids = graph[adr + 2 + nv:adr + 2 + 2 * nv]
    global_faces = graph[adr + 2 + 3 * nv + 3 * nf:end].reshape(nf, 3)
    if np.any(ids < 0) or np.any(ids >= vnum) or len(np.unique(ids)) != nv:
      raise ValueError("compiled collision hull vertex ids are invalid")
    if np.any(global_faces < 0) or np.any(global_faces >= vnum):
      raise ValueError("compiled collision hull face ids are invalid")
    inverse = np.full(vnum, -1, dtype=np.int32)
    inverse[ids] = np.arange(nv, dtype=np.int32)
    faces = inverse[global_faces]
    if np.any(faces < 0):
      raise ValueError("compiled collision hull face references a non-hull vertex")
    return vertices[ids].copy(), faces.copy()

  fadr, nf = int(model.mesh_faceadr[meshid]), int(model.mesh_facenum[meshid])
  faces = np.asarray(model.mesh_face[fadr:fadr + nf], dtype=np.int32)
  if faces.shape != (nf, 3) or nf <= 0 or np.any(faces < 0) or np.any(faces >= vnum):
    raise ValueError("mesh without collision graph has invalid surface faces")
  points = vertices.astype(np.float64)
  interior = points.mean(axis=0)
  for face in faces:
    normal = np.cross(points[face[1]] - points[face[0]],
                      points[face[2]] - points[face[0]])
    length = np.linalg.norm(normal)
    if length < 1e-12:
      continue
    normal /= length
    if normal @ (points[face].mean(axis=0) - interior) < 0:
      normal = -normal
    if np.any((points - points[face[0]]) @ normal > 1e-6):
      raise ValueError("non-convex mesh lacks its compiled collision hull")
  return vertices.copy(), faces.copy()
