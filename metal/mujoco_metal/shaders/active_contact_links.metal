// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

kernel void clear_active_contact_links_masked(
    device int* links [[buffer(0)]],
    device int* overflow [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  int capacity=dims[2], batch=dims[3];
  int width=max(capacity,1), world=int(tid)/width, lane=int(tid)%width;
  if(world>=batch || dims[7+world]==0) return;
  if(lane<capacity) {
    int base=(world*capacity+lane)*2;
    links[base]=-1;
    links[base+1]=-1;
  }
  if(lane==0) overflow[world]=0;
}

kernel void map_active_contact_tree_links(
    device const int* logical_to_packed [[buffer(0)]],
    device const int* pair_contact_offset [[buffer(1)]],
    device const int* pair_trees [[buffer(2)]],
    device const int* equality_active [[buffer(3)]],
    device const int* equality_trees [[buffer(4)]],
    device const int* equality_activity_ids [[buffer(5)]],
    device int* links [[buffer(6)]],
    device int* overflow [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int ncontacts = dims[0];
  int npairs = dims[1];
  int capacity = dims[2];
  int batch = dims[3];
  int neq = dims[4];
  int nequality_links = dims[5];
  bool include_equalities = dims[6] != 0;
  int nlinks = npairs + nequality_links;
  if (nlinks == 0) return;
  if (int(tid) >= batch * max(nlinks, 1)) return;
  int world = int(tid) / max(nlinks, 1);
  int link = int(tid) % max(nlinks, 1);
  if (world >= batch || dims[7+world] == 0) return;
  bool active = false;
  int a = -1;
  int b = -1;
  if (link < npairs) {
    int begin = pair_contact_offset[link];
    int end = pair_contact_offset[link + 1];
    for (int slot = begin; slot < end && slot < ncontacts; ++slot) {
      if (logical_to_packed[world * ncontacts + slot] >= 0) {
        active = true;
        break;
      }
    }
    a = pair_trees[link * 2 + 0];
    b = pair_trees[link * 2 + 1];
  } else {
    int eq_link = link - npairs;
    if (!include_equalities) {
      if (link < capacity) {
        int base = (world * capacity + link) * 2;
        links[base + 0] = -1;
        links[base + 1] = -1;
      }
      return;
    }
    int eq = equality_activity_ids[eq_link];
    active = equality_active[world * neq + eq] != 0;
    a = equality_trees[eq_link * 2 + 0];
    b = equality_trees[eq_link * 2 + 1];
  }
  if (active && a == -2 && b == -2) {
    atomic_store_explicit((device atomic_int*)&overflow[world], 1,
                          memory_order_relaxed);
    return;
  }
  active = active && a >= 0 && b >= 0 && a != b;
  if (link >= capacity) {
    if (active) atomic_store_explicit(
        (device atomic_int*)&overflow[world], 1, memory_order_relaxed);
    return;
  }
  int base = (world * capacity + link) * 2;
  links[base + 0] = -1;
  links[base + 1] = -1;
  if (active) {
    links[base + 0] = a;
    links[base + 1] = b;
  }
}
