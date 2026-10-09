// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;
kernel void reconcile_retained_state(
    device const float* qpos [[buffer(0)]],device const float* qvel [[buffer(1)]],
    device float* words [[buffer(2)]],device float* pos_shadow [[buffer(3)]],
    device float* vel_shadow [[buffer(4)]],constant int* dims [[buffer(5)]],
    uint w [[thread_position_in_grid]]) {
  uint nq=dims[0],nv=dims[1],batch=dims[2],stride=2*nq+2*nv;
  if(w>=batch)return;
  bool pc=(dims[3]&1)!=0,vc=(dims[3]&2)!=0;
  for(uint i=0;i<nq;i++)pc|=as_type<uint>(qpos[w*nq+i])!=as_type<uint>(pos_shadow[w*nq+i]);
  for(uint i=0;i<nv;i++)vc|=as_type<uint>(qvel[w*nv+i])!=as_type<uint>(vel_shadow[w*nv+i]);
  if(pc)for(uint i=0;i<2*nq;i++)words[w*stride+i]=0.0f;
  if(vc)for(uint i=2*nq;i<stride;i++)words[w*stride+i]=0.0f;
  for(uint i=0;i<nq;i++)pos_shadow[w*nq+i]=qpos[w*nq+i];
  for(uint i=0;i<nv;i++)vel_shadow[w*nv+i]=qvel[w*nv+i];
}
