"""CPU-only regression replay for pinned 3.10 flex contact filter semantics.

This compares the actual `filterFlexContacts` state machine (mutable contact
permutation, position-indexed selected/min-distance arrays) with the current
Metal selector's immutable-slot algorithm using identical plane candidate
positions. It intentionally does not replace the native gate.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def _fixture(midphase):
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom type="plane" size="0 0 .1" contype="0" conaffinity="1"/>
      <geom type="plane" pos="0 0 .2" size="0 0 .1"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="8 8 1" pos="0 0 -.1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  if not midphase:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _refresh_source_aabbs(model, data, vertices):
  data.flexvert_xpos[:] = np.asarray(vertices, np.float64)
  child = np.asarray(model.bvh_child, np.int32).reshape(-1, 2)
  nodeid = np.asarray(model.bvh_nodeid, np.int32)
  nbvhstatic = int(model.nbvhstatic)
  for flex in range(int(model.nflex)):
    vbase = int(model.flex_vertadr[flex]); dim = int(model.flex_dim[flex])
    ebase = int(model.flex_elemadr[flex]); edata = int(model.flex_elemdataadr[flex])
    for elem in range(int(model.flex_elemnum[flex])):
      ids = np.asarray(model.flex_elem[edata+elem*(dim+1):edata+(elem+1)*(dim+1)], np.int32)
      pts = data.flexvert_xpos[vbase+ids]; lo=pts.min(0); hi=pts.max(0)
      data.flexelem_aabb[ebase+elem,:3]=.5*(hi+lo)
      data.flexelem_aabb[ebase+elem,3:]=.5*(hi-lo)+float(model.flex_radius[flex])
    bvhadr=int(model.flex_bvhadr[flex]); bvhnum=int(model.flex_bvhnum[flex])
    if bvhadr < 0: continue
    modified=np.zeros(bvhnum,bool)
    for local in range(bvhnum):
      leaf=int(nodeid[bvhadr+local])
      if leaf>=0:
        data.bvh_aabb_dyn[bvhadr+local-nbvhstatic]=data.flexelem_aabb[ebase+leaf]
        modified[local]=True
    for local in range(bvhnum-1,-1,-1):
      if nodeid[bvhadr+local]>=0: continue
      c1,c2=map(int,child[bvhadr+local])
      if not (modified[c1] or modified[c2]): continue
      a=data.bvh_aabb_dyn[bvhadr-nbvhstatic+c1]
      b=data.bvh_aabb_dyn[bvhadr-nbvhstatic+c2]
      lo=np.minimum(a[:3]-a[3:],b[:3]-b[3:]); hi=np.maximum(a[:3]+a[3:],b[:3]+b[3:])
      out=data.bvh_aabb_dyn[bvhadr+local-nbvhstatic]
      out[:3]=.5*(hi+lo); out[3:]=.5*(hi-lo); modified[local]=True


def _source_plane_candidates(model, data, vertices):
  """Mirror pinned mj_collidePlaneFlex input list before filterFlexContacts."""
  result=[]
  radius=float(model.flex_radius[0])
  for geom in range(int(model.ngeom)):
    if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_PLANE): continue
    plane_pos=np.asarray(data.geom_xpos[geom],np.float64)
    mat=np.asarray(data.geom_xmat[geom],np.float64).reshape(9)
    normal=np.array([mat[2],mat[5],mat[8]],np.float64)
    margin=float(model.geom_margin[geom]+model.flex_margin[0])
    gap=float(model.geom_gap[geom]+model.flex_gap[0])
    for vert, v in enumerate(np.asarray(vertices,np.float64)):
      dif=v-plane_pos
      dist=float(dif[0]*normal[0]+dif[1]*normal[1]+dif[2]*normal[2])-radius
      if dist > margin+gap+radius: continue
      pos=v+normal*(-dist*.5-radius)
      result.append({"geom":geom,"vert":vert,"dist":dist,"pos":pos.copy()})
  return result


def _pinned_filter(group, maxkeep=50, dtype=np.float64):
  """Exact mutable-array state machine from engine_collision_driver.c:417."""
  contacts=list(group)
  n=len(contacts)
  if n<=maxkeep: return contacts
  selected=np.zeros(n,dtype=bool)
  min_dist=np.full(n,np.finfo(dtype).max,dtype=dtype)
  best=0; bestval=dtype(-dtype(contacts[0]["dist"]))
  for i in range(1,n):
    val=dtype(-dtype(contacts[i]["dist"]))
    if val>bestval: bestval=val;best=i
  nselected=0
  while nselected<maxkeep and best>=0:
    selected[best]=True
    bestpos=np.asarray(contacts[best]["pos"],dtype=dtype)
    nextbest=-1; nextbestdist=dtype(-1.0)
    for i in range(n):
      if selected[i]: continue
      delta=(np.asarray(contacts[i]["pos"],dtype=dtype)-bestpos).astype(dtype)
      d2=dtype(dtype(delta[0]*delta[0]+delta[1]*delta[1])+dtype(delta[2]*delta[2]))
      if d2<min_dist[i]: min_dist[i]=d2
      if min_dist[i]>nextbestdist: nextbestdist=min_dist[i];nextbest=i
    if nselected<maxkeep-1:
      contacts[nselected],contacts[best]=contacts[best],contacts[nselected]
      if nextbest==nselected: nextbest=best
    nselected+=1;best=nextbest
  return contacts[:nselected]


def _current_shader_immutable(group, maxkeep=50, dtype=np.float32):
  """Mirror flex_contact_select's current fixed-slot group selection."""
  points=np.asarray([c["pos"] for c in group],np.float64).astype(dtype)
  dist=np.asarray([c["dist"] for c in group],np.float64).astype(dtype)
  selected=[]; selected_positions=[]
  for _ in range(min(maxkeep,len(group))):
    best=-1; bestscore=dtype(-np.inf); bestposition=2**31-1
    for slot in range(len(group)):
      if slot in selected: continue
      position=slot
      for pick in range(len(selected)):
        if position==pick: position=selected_positions[pick]
        elif position==selected_positions[pick]: position=pick
      if not selected:
        score=dtype(-dist[slot])
      else:
        score=dtype(np.inf)
        for other in selected:
          delta=(points[slot]-points[other]).astype(dtype)
          value=dtype(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2])
          if value<score: score=value
      if best<0 or score>bestscore or (score==bestscore and position<bestposition):
        best=slot;bestscore=score;bestposition=position
    if best<0: break
    selected.append(best);selected_positions.append(bestposition)
  return [group[i] for i in selected]


def _selector_source_permutation(group, maxkeep=50, score_dtype=np.float64):
  """Replay the candidate MSL permutation and position-indexed state ABI."""
  contacts=list(group)
  n=len(contacts)
  if n<=maxkeep: return contacts
  points=np.asarray([c["pos"] for c in contacts],dtype=np.float64)
  distances=np.asarray([c["dist"] for c in contacts],dtype=np.float64)
  permutation=list(range(n))
  selected=np.zeros(n,dtype=np.uint8)
  min_dist=np.full(n,1.0e10,dtype=score_dtype)  # pinned mjMAXVAL
  best=0; bestdist=score_dtype(-score_dtype(distances[0]))
  for i in range(1,n):
    candidate=score_dtype(-score_dtype(distances[i]))
    if candidate>bestdist: bestdist,best=candidate,i
  nselected=0
  while nselected<maxkeep and best>=0:
    selected[best]=1
    bestpos=points[permutation[best]]
    nextbest=-1; nextbestdist=score_dtype(-1.0)
    for i in range(n):
      if selected[i]: continue
      delta=np.asarray(points[permutation[i]]-bestpos,dtype=score_dtype)
      d2=score_dtype(score_dtype(delta[0]*delta[0]+delta[1]*delta[1])
                     +score_dtype(delta[2]*delta[2]))
      if d2<min_dist[i]: min_dist[i]=d2
      if min_dist[i]>nextbestdist: nextbestdist,nextbest=min_dist[i],i
    if nselected<maxkeep-1:
      permutation[nselected],permutation[best]=(
          permutation[best],permutation[nselected])
      if nextbest==nselected: nextbest=best
    nselected+=1;best=nextbest
  return [contacts[i] for i in permutation[:nselected]]


@pytest.mark.parametrize("midphase", [False, True])
def test_cpu_literal_filter_state_machine_matches_installed_pinned_driver(midphase):
  model,data=_fixture(midphase)
  exact=np.asarray(data.flexvert_xpos,np.float64).copy()
  high=exact.astype(np.float32)
  expected_data=mujoco.MjData(model); expected_data.qpos[:]=data.qpos; expected_data.qvel[:]=data.qvel
  mujoco.mj_forward(model,expected_data);_refresh_source_aabbs(model,expected_data,high);mujoco.mj_collision(model,expected_data)
  expected={(int(c.geom[0]),int(c.vert[1])) for c in expected_data.contact[:expected_data.ncon]}
  candidates=_source_plane_candidates(model,expected_data,high)
  groups=( [candidates] if midphase else
      [[c for c in candidates if c["geom"]==g] for g in (0,1)] )
  source={ (c["geom"],c["vert"]) for group in groups for c in _pinned_filter(group) }
  assert source==expected
  assert len(source)==expected_data.ncon


@pytest.mark.parametrize("geom", [0,1])
def test_cpu_current_selector_semantics_are_not_pinned_mutable_filter(geom):
  model,data=_fixture(False)
  high=np.asarray(data.flexvert_xpos,np.float64).astype(np.float32)
  oracle=mujoco.MjData(model);oracle.qpos[:]=data.qpos;oracle.qvel[:]=data.qvel
  mujoco.mj_forward(model,oracle);_refresh_source_aabbs(model,oracle,high);mujoco.mj_collision(model,oracle)
  expected={(int(c.geom[0]),int(c.vert[1])) for c in oracle.contact[:oracle.ncon] if int(c.geom[0])==geom}
  group=[c for c in _source_plane_candidates(model,oracle,high) if c["geom"]==geom]
  source={(c["geom"],c["vert"]) for c in _pinned_filter(group)}
  current={(c["geom"],c["vert"]) for c in _current_shader_immutable(group,dtype=np.float64)}
  assert source==expected
  assert current != expected
  assert {v for g,v in source^current}=={3,12,14,17,18,30}


def test_cpu_float32_score_also_loses_pinned_strict_distance_order():
  model,data=_fixture(False)
  high=np.asarray(data.flexvert_xpos,np.float64).astype(np.float32)
  oracle=mujoco.MjData(model);oracle.qpos[:]=data.qpos;oracle.qvel[:]=data.qvel
  mujoco.mj_forward(model,oracle);_refresh_source_aabbs(model,oracle,high);mujoco.mj_collision(model,oracle)
  expected={(int(c.geom[0]),int(c.vert[1])) for c in oracle.contact[:oracle.ncon] if int(c.geom[0])==0}
  group=[c for c in _source_plane_candidates(model,oracle,high) if c["geom"]==0]
  f32={(c["geom"],c["vert"]) for c in _pinned_filter(group,dtype=np.float32)}
  source={(c["geom"],c["vert"]) for c in _pinned_filter(group,dtype=np.float64)}
  assert source==expected
  assert {v for g,v in source^f32}=={9,17,22,23}


@pytest.mark.parametrize("midphase", [False, True])
def test_position_indexed_selector_replays_original_pinned_plane_fixture(midphase):
  model,data=_fixture(midphase)
  high=np.asarray(data.flexvert_xpos,np.float64).astype(np.float32)
  oracle=mujoco.MjData(model);oracle.qpos[:]=data.qpos;oracle.qvel[:]=data.qvel
  mujoco.mj_forward(model,oracle);_refresh_source_aabbs(model,oracle,high)
  mujoco.mj_collision(model,oracle)
  expected={(int(c.geom[0]),int(c.vert[1]))
            for c in oracle.contact[:oracle.ncon]}
  candidates=_source_plane_candidates(model,oracle,high)
  groups=([candidates] if midphase else
          [[c for c in candidates if c["geom"]==geom] for geom in (0,1)])
  selected=[contact for group in groups
            for contact in _selector_source_permutation(group)]
  actual={(c["geom"],c["vert"]) for c in selected}
  assert actual==expected
  assert len(actual)==oracle.ncon
