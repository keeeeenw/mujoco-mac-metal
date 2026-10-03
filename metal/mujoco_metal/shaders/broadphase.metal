// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Milestone 017: explicit scalable broadphase. Conservative per-pair sphere
// overlap mask (deterministic pair order, no reordering) written before the
// narrowphase. The contact kernel keeps its own early-outs (behavior
// unchanged); this mask is the explicit, counted broadphase used for
// capacity diagnostics and peak-active reporting. Reads (not writes) happen
// on the host only through diagnostics APIs, never inside the step loop.

#include <metal_stdlib>
using namespace metal;

// Conservative overlap: center distance <= rbound sum + margin + gap.
// Epsilon keeps the mask a superset of narrowphase hits at float boundary.
kernel void broadphase_mask(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_rbound [[buffer(1)]],
    device const int* pair_geoms [[buffer(2)]],
    device const float* pair_margin_gap [[buffer(3)]],
    device const int* geom_type [[buffer(4)]],
    device float* mask_out [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  // dims reuse c_dims: [nv, npairs, ncontacts_max, batch, nbody, njnt,
  //   ngeom, cone, disable_multiccd]. Only npairs/batch/ngeom are read.
  int npairs = dims[1];
  int batch = dims[3];
  int ngeom = dims[6];
  int world = int(tid) / max(npairs, 1);
  int pair_idx = int(tid) % max(npairs, 1);
  if (uint(world) >= uint(batch) || pair_idx >= npairs) return;
  int a = pair_geoms[pair_idx * 2 + 0];
  int b = pair_geoms[pair_idx * 2 + 1];
  // Planes (type 0) are unbounded: always overlap (conservative).
  if (geom_type[a] == 0 || geom_type[b] == 0) {
    mask_out[world * npairs + pair_idx] = 1.0f;
    return;
  }
  int go = world * ngeom;
  float3 pa = float3(geom_pos[(go + a) * 3], geom_pos[(go + a) * 3 + 1], geom_pos[(go + a) * 3 + 2]);
  float3 pb = float3(geom_pos[(go + b) * 3], geom_pos[(go + b) * 3 + 1], geom_pos[(go + b) * 3 + 2]);
  float R = geom_rbound[a] + geom_rbound[b]
      + pair_margin_gap[pair_idx * 2 + 0] + pair_margin_gap[pair_idx * 2 + 1];
  float d = length(pa - pb);
  mask_out[world * npairs + pair_idx] = (d <= R + 1e-6f) ? 1.0f : 0.0f;
}
