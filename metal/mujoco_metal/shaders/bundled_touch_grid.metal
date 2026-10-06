// MuJoCo 3.10.0 bundled mujoco.sensor.touch_grid.
#include <metal_stdlib>
using namespace metal;

static inline float3 cross3(float3 a, float3 b) {
  return float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

static inline float3 rotate_q(float4 q, float3 v) {
  float3 u = q.yzw;
  float3 t = 2.0f * cross3(u, v);
  return v + q.x*t + cross3(u, t);
}

static inline int lower_bound_edge(device const float* edges, int base,
                                   int count, float value) {
  int lo = 0, hi = count;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (value <= edges[base + mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

kernel void bundled_touch_grid(
    device const int* contact_valid [[buffer(0)]],
    device const float* contact_frame [[buffer(1)]],
    device const float* contact_force [[buffer(2)]],
    device const int* slot_pair [[buffer(3)]],
    device const int* pair_geoms [[buffer(4)]],
    device const int* geom_body [[buffer(5)]],
    device const int* body_weld [[buffer(6)]],
    device const float* site_pos [[buffer(7)]],
    device const float* site_quat [[buffer(8)]],
    device const int* sensor_meta [[buffer(9)]],
    device const float* sensor_params [[buffer(10)]],
    device const float* x_edges [[buffer(11)]],
    device const float* y_edges [[buffer(12)]],
    device const bool* compute_world [[buffer(13)]],
    device float* sensordata [[buffer(14)]],
    constant int* dims [[buffer(15)]],
    device const int* sensor_index [[buffer(16)]],
    device const int* contact_condim [[buffer(17)]],
    device const float* contact_friction [[buffer(18)]],
    device int* world_status [[buffer(19)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], nc = dims[1], ngeom = dims[2], nbody = dims[3];
  int nsensor = dims[4], edge_stride = dims[5], npair = dims[9];
  if (nsensor <= 0) return;
  int si = sensor_index[0];
  if (si < 0 || si >= nsensor) return;
  int site = sensor_meta[8*si + 0];
  int address = sensor_meta[8*si + 1];
  int dimension = sensor_meta[8*si + 2];
  int nchannel = sensor_meta[8*si + 3];
  int sx = sensor_meta[8*si + 4], sy = sensor_meta[8*si + 5];
  int parent_body = sensor_meta[8*si + 6];
  int parent_weld = sensor_meta[8*si + 7];
  int per_world = dimension;
  uint total = uint(batch * per_world);
  if (tid >= total) return;
  int world = int(tid / uint(per_world));
  if (!compute_world[world]) return;
  int local = int(tid % uint(per_world));
  int frame_size = sx * sy;
  int channel = local / frame_size;
  int pixel = local - channel * frame_size;
  int ix = pixel % sx, iy = pixel / sx;
  float fx = 0.0f;
  // Site quaternion is laid out [B, nsite, 4] in wxyz order.
  int nsite = dims[6];
  uint sbase = uint((world*nsite + site)*4);
  float4 sq = float4(site_quat[sbase], site_quat[sbase+1],
                     site_quat[sbase+2], site_quat[sbase+3]);
  float4 invq = float4(sq.x, -sq.y, -sq.z, -sq.w);
  uint pbase = uint((world*nsite + site)*3);
  float3 sp = float3(site_pos[pbase], site_pos[pbase+1], site_pos[pbase+2]);
  bool invalid = !all(isfinite(sq)) || !all(isfinite(sp));
  int edge_base = si * edge_stride;
  for (int c = 0; c < nc; ++c) {
    uint cb = uint(world*nc + c);
    if (contact_valid[cb] <= 0) continue;
    int pair = slot_pair[c];
    if (pair == -1) continue;
    // slot_pair is generated from a fixed contact workspace. Treat an
    // out-of-range pair as a malformed source row rather than indexing the
    // model-static pair table with untrusted device data.
    if (pair < 0 || pair >= npair) {
      invalid = true;
      continue;
    }
    int g1 = pair_geoms[2*pair], g2 = pair_geoms[2*pair+1];
    if (g1 < 0 || g1 >= ngeom || g2 < 0 || g2 >= ngeom) continue;
    int b1 = geom_body[g1], b2 = geom_body[g2];
    if (b1 < 0 || b1 >= nbody || b2 < 0 || b2 >= nbody) continue;
    if (body_weld[b1] != parent_weld && body_weld[b2] != parent_weld) continue;
    uint fb = cb * 12;
    float3 n = float3(contact_frame[fb], contact_frame[fb+1],
                      contact_frame[fb+2]);
    float3 t1 = float3(contact_frame[fb+3], contact_frame[fb+4],
                       contact_frame[fb+5]);
    float3 t2 = float3(contact_frame[fb+6], contact_frame[fb+7],
                       contact_frame[fb+8]);
    int cdim = contact_condim[c*3];
    float3 cp = float3(contact_frame[fb+9], contact_frame[fb+10],
                       contact_frame[fb+11]);
    if (!all(isfinite(n)) || !all(isfinite(t1)) || !all(isfinite(t2))
        || !all(isfinite(cp))) {
      invalid = true;
      continue;
    }
    float wrench[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (dims[8] == 1) {
      for (int k = 0; k < min(cdim, 6); ++k) {
        wrench[k] = contact_force[cb*11 + k];
        if (!isfinite(wrench[k])) invalid = true;
      }
    } else {
      wrench[0] = contact_force[cb*11];
      if (!isfinite(wrench[0])) invalid = true;
      int nfric = min(max(cdim - 1, 0), 5);
      for (int k = 0; k < nfric; ++k) {
        float plus = contact_force[cb*11 + 1 + 2*k];
        float minus = contact_force[cb*11 + 2 + 2*k];
        float friction = contact_friction[c*5 + k];
        wrench[k+1] = friction * (plus - minus);
        if (!isfinite(plus) || !isfinite(minus) || !isfinite(friction)
            || !isfinite(wrench[k+1])) invalid = true;
      }
    }
    if (invalid) continue;
    float3 f = n*wrench[0] + t1*wrench[1] + t2*wrench[2];
    float3 torque = n*wrench[3] + t1*wrench[4] + t2*wrench[5];
    if (parent_body < max(b1, b2)) { f = -f; torque = -torque; }
    float3 localf = rotate_q(invq, f);
    float3 localt = rotate_q(invq, torque);
    float value = channel < 3 ? (channel == 0 ? localf.z :
                  (channel == 1 ? localf.x : localf.y)) :
                  (channel == 3 ? localt.z : (channel == 4 ? localt.x : localt.y));
    float3 xyz = rotate_q(invq, cp-sp);
    if (!all(isfinite(xyz)) || !isfinite(value)) {
      invalid = true;
      continue;
    }
    float azimuth = atan2(xyz.x, -xyz.z);
    float elevation = atan2(xyz.y, sqrt(xyz.x*xyz.x + xyz.z*xyz.z));
    int ex = lower_bound_edge(x_edges, edge_base, sx+1, azimuth);
    int ey = lower_bound_edge(y_edges, edge_base, sy+1, elevation);
    if (ex == 0 || ex == sx+1 || ey == 0 || ey == sy+1) continue;
    if (ex-1 == ix && ey-1 == iy) {
      fx += value;
    }
  }
  if (invalid) {
    fx = 0.0f;
    if (local == 0) world_status[world] = 3;
  }
  uint output = uint(world*int(dims[7]) + address + local);
  if (channel < nchannel) sensordata[output] = fx;
}
