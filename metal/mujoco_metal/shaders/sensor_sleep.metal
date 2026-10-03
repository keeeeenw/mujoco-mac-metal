#include <metal_stdlib>
using namespace metal;

kernel void sensor_awake_mask(
    device const int* tree_awake [[buffer(0)]],
    device const int* object_treeids [[buffer(1)]],
    device const int* object_counts [[buffer(2)]],
    device const int* always_awake [[buffer(3)]],
    device const int* sensor_refs [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    device int* output [[buffer(6)]],
    uint world [[thread_position_in_grid]]) {
  uint batch = uint(dims[0]);
  uint nsensor = uint(dims[1]);
  uint ntree = uint(dims[2]);
  if (world >= batch) return;

  // Sensor objects are valid pinned mjOBJ references. Their sleep state is
  // recursively derived from the referenced sensor, so resolve dependencies
  // in-place. Each pass propagates one level; a DAG settles in at most
  // nsensor passes. Cycles retain the awake initialization (fail open).
  for (uint sensor = 0; sensor < nsensor; ++sensor) {
    output[world * nsensor + sensor] = 1;
  }
  for (uint pass = 0; pass < nsensor; ++pass) {
    bool changed = false;
    for (uint sensor = 0; sensor < nsensor; ++sensor) {
      bool object_awake[2] = {false, false};
      bool object_asleep[2] = {false, false};
      bool object_unknown[2] = {false, false};
      for (uint object = 0; object < 2; ++object) {
        int count = object_counts[sensor * 2 + object];
        if (count == -1) {
          object_unknown[object] = true;  // mjOBJ_UNKNOWN is mjS_AWAKE.
        } else if (count == -2) {
          object_asleep[object] = true;
        } else if (count == -3) {
          object_awake[object] = true;
        } else if (count == -4) {
          int target = sensor_refs[sensor * 2 + object];
          if (target >= 0 && uint(target) < nsensor) {
            object_awake[object] =
                output[world * nsensor + uint(target)] != 0;
            object_asleep[object] = !object_awake[object];
          } else {
            object_awake[object] = true;
          }
        } else if (count > 0) {
          for (int k = 0; k < count; ++k) {
            int tree = object_treeids[(sensor * 2 + object) * 2 + uint(k)];
            if (tree >= 0 && uint(tree) < ntree
                && tree_awake[world * ntree + uint(tree)] != 0) {
              object_awake[object] = true;
            }
          }
          object_asleep[object] = !object_awake[object];
        }
      }
      bool awake;
      if (always_awake[sensor]) {
        awake = true;
      } else if (object_unknown[0] && object_unknown[1]) {
        awake = true;
      } else if (object_unknown[0]) {
        awake = !object_asleep[1];
      } else if (object_unknown[1]) {
        awake = !object_asleep[0];
      } else {
        awake = object_awake[0] || object_awake[1];
      }
      uint out_index = world * nsensor + sensor;
      int next = int(awake);
      changed = changed || output[out_index] != next;
      output[out_index] = next;
    }
    if (!changed) break;
  }
}
