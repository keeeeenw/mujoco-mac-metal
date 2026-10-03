// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Stable fixed-capacity flag compaction. A block-local inclusive scan runs
// in parallel, block sums are scanned per world, then a parallel scatter
// emits deterministic maps. No dynamic-size allocation or host count
// readback is needed, and an overflow world emits no partial map.

#include <metal_stdlib>
using namespace metal;

constexpr int kScanBlockSize = 256;

kernel void compact_flag_blocks(
    device const float* flags [[buffer(0)]],
    device int* local_prefix [[buffer(1)]],
    device int* block_sum [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  int logical_count = dims[0];
  int nblocks = dims[3];
  int world = int(group) / nblocks;
  if (world >= dims[2]) return;
  int block = int(group) % nblocks;
  int logical = block * kScanBlockSize + int(lane);
  int base = world * logical_count;
  threadgroup int scan[kScanBlockSize];
  scan[lane] = (logical < logical_count && flags[base + logical] != 0.0f) ? 1 : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int offset = 1; offset < kScanBlockSize; offset <<= 1) {
    int add = (int(lane) >= offset) ? scan[lane - offset] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[lane] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (logical < logical_count) local_prefix[base + logical] = scan[lane];
  if (int(lane) == kScanBlockSize - 1) {
    block_sum[world * nblocks + block] = scan[lane];
  }
}

kernel void scan_compaction_blocks(
    device const int* block_sum [[buffer(0)]],
    device int* block_offset [[buffer(1)]],
    device int* active_count [[buffer(2)]],
    device int* overflow [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint world [[thread_position_in_grid]]) {
  int capacity = dims[1];
  int batch = dims[2];
  int nblocks = dims[3];
  if (int(world) >= batch) return;
  int total = 0;
  int base = int(world) * nblocks;
  for (int block = 0; block < nblocks; ++block) {
    block_offset[base + block] = total;
    total += block_sum[base + block];
  }
  active_count[world] = total;
  overflow[world] = (total > capacity) ? 1 : 0;
}

kernel void scatter_compaction_maps(
    device const float* flags [[buffer(0)]],
    device const int* local_prefix [[buffer(1)]],
    device const int* block_offset [[buffer(2)]],
    device const int* overflow [[buffer(3)]],
    device int* packed_to_logical [[buffer(4)]],
    device int* logical_to_packed [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int logical_count = dims[0];
  int capacity = dims[1];
  int nblocks = dims[3];
  int total_threads = int(tid);
  int world = total_threads / max(logical_count, 1);
  int logical = total_threads % max(logical_count, 1);
  if (world >= dims[2]) return;
  if (logical_count == 0 || logical >= logical_count || overflow[world]) return;
  int base = world * logical_count;
  if (flags[base + logical] == 0.0f) return;
  int block = logical / kScanBlockSize;
  int packed = block_offset[world * nblocks + block] + local_prefix[base + logical] - 1;
  if (packed >= capacity) return;
  packed_to_logical[world * capacity + packed] = logical;
  logical_to_packed[base + logical] = packed;
}
