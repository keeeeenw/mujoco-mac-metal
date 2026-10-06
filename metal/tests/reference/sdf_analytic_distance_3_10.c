// Copyright 2021 DeepMind Technologies Limited
// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Pinned MuJoCo 3.10.0 engine_collision_sdf.c radialField3d/geomDistance.
// Function bodies retained verbatim. Only analytic types enter this wrapper.
#include <mujoco/mujoco.h>
#define mjERROR mju_error
static mjtNum oct_distance(const mjModel* m, const mjtNum x[3], int i) {
  mju_error("octree outside analytic reference scope"); return 0;
}
static void radialField3d(mjtNum field[3], const mjtNum a[3], const mjtNum x[3],
                          const mjtNum size[3]) {
  field[0] = -size[0] / a[0];
  field[1] = -size[1] / a[1];
  field[2] = -size[2] / a[2];
  mju_normalize3(field);

  // flip sign if necessary
  if (x[0] < 0) field[0] = -field[0];
  if (x[1] < 0) field[1] = -field[1];
  if (x[2] < 0) field[2] = -field[2];
}

static mjtNum geomDistance(const mjModel* m, const mjData* d, const mjpPlugin* p,
                           int i, const mjtNum x[3], mjtGeom type) {
  mjtNum a[3], b[3];
  const mjtNum* size = m->geom_size+3*i;

  // see https://iquilezles.org/articles/distfunctions/
  switch (type) {
  case mjGEOM_PLANE:
    return x[2];

  case mjGEOM_SPHERE:
    return mju_norm3(x) - size[0];

  case mjGEOM_BOX:
    // compute shortest distance to box surface if outside, otherwise
    // intersect with a unit gradient that linearly rotates from radial to the face normals
    a[0] = mju_abs(x[0]) - size[0];
    a[1] = mju_abs(x[1]) - size[1];
    a[2] = mju_abs(x[2]) - size[2];
    if (a[0] >= 0 || a[1] >= 0 || a[2] >= 0) {
      b[0] = mju_max(a[0], 0);
      b[1] = mju_max(a[1], 0);
      b[2] = mju_max(a[2], 0);
      return mju_norm3(b) + mju_min(mju_max(a[0], mju_max(a[1], a[2])), 0);
    }
    radialField3d(b, a, x, size);
    mjtNum t[3];
    t[0] = -a[0] / mju_abs(b[0]);
    t[1] = -a[1] / mju_abs(b[1]);
    t[2] = -a[2] / mju_abs(b[2]);
    return -mju_min(t[0], mju_min(t[1], t[2])) * mju_norm3(b);

  case mjGEOM_CAPSULE:
    a[0] = x[0];
    a[1] = x[1];
    a[2] = x[2] - mju_clip(x[2], -size[1], size[1]);
    return mju_norm3(a) - size[0];

  case mjGEOM_ELLIPSOID:
    a[0] = x[0] / size[0];
    a[1] = x[1] / size[1];
    a[2] = x[2] / size[2];
    b[0] = a[0] / size[0];
    b[1] = a[1] / size[1];
    b[2] = a[2] / size[2];
    mjtNum k0 = mju_norm3(a);
    mjtNum k1 = mju_norm3(b);
    return k0 * (k0 - 1.0) / k1;

  case mjGEOM_CYLINDER:
    a[0] = mju_sqrt(x[0]*x[0]+x[1]*x[1]) - size[0];
    a[1] = mju_abs(x[2]) - size[1];
    b[0] = mju_max(a[0], 0);
    b[1] = mju_max(a[1], 0);
    return mju_min(mju_max(a[0], a[1]), 0) + mju_norm(b, 2);

  case mjGEOM_SDF:
    if (p) {
      return p->sdf_distance(x, d, i);
    } else {
      return oct_distance(m, x, i);
    }

  case mjGEOM_MESH:
    if (m->mesh_octadr[i] == -1) {
      mjERROR("sdf queries require needsdf=\"true\" on mesh %d", i);
      return 0;
    }
    return oct_distance(m, x, i);

  default:
    mjERROR("sdf collisions not available for geom type %d", type);
    return 0;
  }
}


__attribute__((visibility("default")))
mjtNum source_analytic_distance(int type, const mjtNum size[3], const mjtNum x[3]) {
  if (type != mjGEOM_PLANE && type != mjGEOM_SPHERE && type != mjGEOM_CAPSULE &&
      type != mjGEOM_BOX && type != mjGEOM_ELLIPSOID && type != mjGEOM_CYLINDER) {
    mju_error("unsupported analytic reference type"); return 0;
  }
  mjModel model = {0};
  model.geom_size = (mjtNum*)size;
  return geomDistance(&model, NULL, NULL, 0, x, (mjtGeom)type);
}
