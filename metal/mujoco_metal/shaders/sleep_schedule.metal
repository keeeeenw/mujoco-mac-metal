// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Device-side MuJoCo 3.10 tree sleep scheduling. tree_state is the pinned
// signed tree_asleep representation: negative awake countdowns and, for a
// sleeping island, a closed cycle of tree indices. No state is read back in
// the stepping path.

#include <metal_stdlib>
using namespace metal;

inline bool wake_sleep_cycle(device int* state, int base, int ntree,
                             int tree, int wake_value) {
  if (tree < 0 || tree >= ntree) return false;
  if (state[base + tree] < 0) {
    state[base + tree] = min(wake_value, state[base + tree]);
    return true;
  }
  int current = tree;
  for (int count = 0; count < ntree; ++count) {
    int next = state[base + current];
    if (next < 0 || next >= ntree) return false;
    state[base + current] = wake_value;
    current = next;
    if (current == tree) return true;
  }
  return false;
}

inline bool same_sleep_cycle(device const int* state, int base, int ntree,
                             int tree, int target) {
  if (tree < 0 || tree >= ntree || target < 0 || target >= ntree) return false;
  int current = tree;
  for (int count = 0; count < ntree; ++count) {
    if (current == target) return true;
    int next = state[base + current];
    if (next < 0 || next >= ntree) return false;
    current = next;
    if (current == tree) return false;
  }
  return false;
}

kernel void sleep_transition(
    device const float* qvel [[buffer(0)]],
    device const float* qfrc_applied [[buffer(1)]],
    device const float* xfrc_applied [[buffer(2)]],
    device const float* dof_length [[buffer(3)]],
    device const int* tree_dofadr [[buffer(4)]],
    device const int* tree_dofnum [[buffer(5)]],
    device const int* tree_bodyadr [[buffer(6)]],
    device const int* tree_bodynum [[buffer(7)]],
    device const int* tree_policy [[buffer(8)]],
    device const int* tree_links [[buffer(9)]],
    device const int* constraints_active [[buffer(10)]],
    device const float* sleep_tolerance [[buffer(11)]],
    device int* tree_state [[buffer(12)]],
    device int* tree_awake [[buffer(13)]],
    device int* sleep_status [[buffer(14)]],
    device const int* links_overflow [[buffer(15)]],
    constant int* dims [[buffer(16)]],
    constant int* advance_mode [[buffer(17)]],
    device const int* equality_active [[buffer(18)]],
    device const int* equality_kind [[buffer(19)]],
    device const int* equality_pair_trees [[buffer(20)]],
    device const int* flex_offsets [[buffer(21)]],
    device const int* flex_tree_ids [[buffer(22)]],
    device const int* active_worlds [[buffer(23)]],
    device const int* pose_mismatch [[buffer(24)]],
    uint world [[thread_position_in_grid]]) {
  int nv = dims[0];
  int nbody = dims[1];
  int ntree = dims[3];
  int max_links = dims[4];
  int batch = dims[5];
  bool enabled = dims[6] != 0;
  bool island_disabled = dims[7] != 0;
  int neq = dims[8];
  int nflex_members = dims[9];
  if (int(world) >= batch) return;
  // Failed/disabled physical rows must retain every scheduler-owned value.
  // Return before status, tree state, or awake bits are written.
  if (active_worlds[world] == 0) return;
  int state_base = int(world) * max(ntree, 1);
  int qvel_base = int(world) * max(nv, 1);
  int body_base = int(world) * nbody * 6;
  sleep_status[world] = links_overflow[world] ? 2 : 0;
  if (sleep_status[world] != 0) {
    for (int tree = 0; tree < ntree; ++tree) {
      tree_state[state_base + tree] = -11;
      tree_awake[state_base + tree] = 1;
    }
    return;
  }
  if (!enabled || (island_disabled && constraints_active[world] != 0)) {
    for (int tree = 0; tree < ntree; ++tree) {
      tree_state[state_base + tree] = -11;
      tree_awake[state_base + tree] = 1;
    }
    return;
  }

  // mj_kinematics1 marks an articulated sleeping tree awake when its current
  // candidate xpos/xquat differs from the retained position-stage cache.
  // mj_wake then wakes the complete pinned sleep cycle before kinematics2,
  // collision, or constraint assembly. The host supplies this device-produced
  // mismatch mask only on the pre-solve wake pass.
  if (advance_mode[0] == 0) {
    for (int tree = 0; tree < ntree; ++tree) {
      if (pose_mismatch[state_base + tree] != 0
          && tree_state[state_base + tree] >= 0) {
        if (!wake_sleep_cycle(tree_state, state_base, ntree, tree, -11)) {
          sleep_status[world] = 1;
          return;
        }
      }
    }
  }

  float tolerance = sleep_tolerance[0];
  for (int tree = 0; tree < ntree; ++tree) {
    int state = tree_state[state_base + tree];
    bool can_sleep = tree_policy[tree] != 1 && tree_policy[tree] != 3;
    bool perturbed = !can_sleep;
    int dof_adr = tree_dofadr[tree];
    int dof_count = tree_dofnum[tree];
    float weighted_speed = 0.0f;
    bool zero_velocity = true;
    for (int j = 0; j < dof_count; ++j) {
      int dof = dof_adr + j;
      weighted_speed = max(weighted_speed,
          fabs(qvel[qvel_base + dof]) * dof_length[dof]);
      // mj_wake calls treeCanSleep(..., 0), which checks qvel bytes for exact
      // zero. The configured tolerance belongs to mj_sleep, not this wake
      // pass. Keep the two tests separate: a small nonzero velocity can be
      // sleep-eligible and still must wake an already sleeping tree.
      zero_velocity = zero_velocity && qvel[qvel_base + dof] == 0.0f;
      can_sleep = can_sleep && qfrc_applied[qvel_base + dof] == 0.0f;
      perturbed = perturbed || qfrc_applied[qvel_base + dof] != 0.0f
          || qvel[qvel_base + dof] != 0.0f;
    }
    if (advance_mode[0] == 0) {
      can_sleep = can_sleep && zero_velocity;
    } else {
      can_sleep = can_sleep && (tolerance > 0.0f
          ? weighted_speed < tolerance : weighted_speed == 0.0f);
    }
    int body_adr = tree_bodyadr[tree];
    int body_count = tree_bodynum[tree];
    for (int j = 0; j < body_count; ++j) {
      int body = body_adr + j;
      for (int axis = 0; axis < 6; ++axis) {
        can_sleep = can_sleep && xfrc_applied[body_base + body * 6 + axis] == 0.0f;
        perturbed = perturbed || xfrc_applied[body_base + body * 6 + axis] != 0.0f;
      }
    }
    if (advance_mode[0] == 0) {
      // mj_wake only sweeps trees that are already asleep. Awake countdowns
      // are left untouched here; mj_sleep evaluates those with the configured
      // tolerance after activation advance. Resetting an awake countdown for
      // any nonzero qvel prevents quiet trees from ever reaching sleep.
      if (state >= 0 && perturbed
          && !wake_sleep_cycle(tree_state, state_base, ntree, tree, -11)) {
        sleep_status[world] = 1;
      }
    } else if (state < 0 && !can_sleep) {
      // mj_sleep skips trees which are already asleep.
      tree_state[state_base + tree] = -11;
    }
  }

  // Existing sleeping cycles wake as a unit if their eligibility was lost.
  if (advance_mode[0] == 0 && sleep_status[world] == 0) {
    for (int tree = 0; tree < ntree; ++tree) {
      int state = tree_state[state_base + tree];
      if (state >= 0) {
        int dof_adr = tree_dofadr[tree];
        int dof_count = tree_dofnum[tree];
        bool can_sleep = tree_policy[tree] != 1 && tree_policy[tree] != 3;
        bool perturbed = !can_sleep;
        float weighted_speed = 0.0f;
        for (int j = 0; j < dof_count; ++j) {
          int dof = dof_adr + j;
          weighted_speed = max(weighted_speed,
              fabs(qvel[qvel_base + dof]) * dof_length[dof]);
          can_sleep = can_sleep && qfrc_applied[qvel_base + dof] == 0.0f;
          perturbed = perturbed || qfrc_applied[qvel_base + dof] != 0.0f
              || qvel[qvel_base + dof] != 0.0f;
        }
        can_sleep = can_sleep && (tolerance > 0.0f
            ? weighted_speed < tolerance : weighted_speed == 0.0f);
        int body_adr = tree_bodyadr[tree];
        int body_count = tree_bodynum[tree];
        for (int j = 0; j < body_count; ++j) {
          int body = body_adr + j;
          for (int axis = 0; axis < 6; ++axis) {
            can_sleep = can_sleep && xfrc_applied[body_base + body * 6 + axis] == 0.0f;
            perturbed = perturbed || xfrc_applied[body_base + body * 6 + axis] != 0.0f;
          }
        }
        if (perturbed && !wake_sleep_cycle(tree_state, state_base, ntree, tree, -11)) {
          sleep_status[world] = 1;
          break;
        }
      }
    }
  }

  // Actual contacts use mj_wakeCollision's rule: a contact against a static
  // tree is ignored; exactly one sleeping island wakes to its awake partner's
  // countdown. Two sleeping contact trees are an invalid MuJoCo state.
  for (int pass = 0; pass < ntree && sleep_status[world] == 0; ++pass) {
    bool changed = false;
    for (int link = 0; link < max_links; ++link) {
      int a = tree_links[(int(world) * max_links + link) * 2 + 0];
      int b = tree_links[(int(world) * max_links + link) * 2 + 1];
      if (a == -1 && b == -1) continue;
      if (a < 0 || b < 0 || a >= ntree || b >= ntree || a == b) {
        sleep_status[world] = 1;
        break;
      }
      int sa = tree_state[state_base + a];
      int sb = tree_state[state_base + b];
      if (sa >= 0 && sb < 0) {
        if (!wake_sleep_cycle(tree_state, state_base, ntree, a, sb)) {
          sleep_status[world] = 1;
          break;
        }
        changed = true;
      } else if (sb >= 0 && sa < 0) {
        if (!wake_sleep_cycle(tree_state, state_base, ntree, b, sa)) {
          sleep_status[world] = 1;
          break;
        }
        changed = true;
      } else if (sa >= 0 && sb >= 0) {
        // Pinned mj_wakeCollision treats a contact between two sleepers as
        // an invariant violation, even when their cycle encoding is shared.
        sleep_status[world] = 3;
        break;
      }
    }
    if (!changed) break;
  }

  // Match mj_wakeEquality's separate, model-ordered sweep. Pair constraints
  // wake a sleeping island fully; flex equalities scan member body order,
  // select the first awake tree, then wake only the first sleeping island to
  // that awake tree's current countdown. Tendon equality is rejected during
  // lowering because pinned MuJoCo raises an error for it.
  for (int eq = 0; eq < neq && sleep_status[world] == 0; ++eq) {
    if (equality_active[world * neq + eq] == 0) continue;
    int kind = equality_kind[eq];
    if (kind == 1) {
      int a = equality_pair_trees[eq * 2 + 0];
      int b = equality_pair_trees[eq * 2 + 1];
      if (a < 0 || b < 0 || a == b) continue;
      if (a >= ntree || b >= ntree) { sleep_status[world] = 1; break; }
      int sa = tree_state[state_base + a];
      int sb = tree_state[state_base + b];
      if (sa >= 0 && sb >= 0) {
        if (!same_sleep_cycle(tree_state, state_base, ntree, a, b)) {
          if (!wake_sleep_cycle(tree_state, state_base, ntree, a, -11)
              || !wake_sleep_cycle(tree_state, state_base, ntree, b, -11)) {
            sleep_status[world] = 1;
          }
        }
      } else if (sa >= 0) {
        if (!wake_sleep_cycle(tree_state, state_base, ntree, a, -11))
          sleep_status[world] = 1;
      } else if (sb >= 0) {
        if (!wake_sleep_cycle(tree_state, state_base, ntree, b, -11))
          sleep_status[world] = 1;
      }
    } else if (kind == 2) {
      int begin = flex_offsets[eq];
      int end = flex_offsets[eq + 1];
      if (begin < 0 || end < begin || end > nflex_members) {
        sleep_status[world] = 1;
        break;
      }
      int awake_tree = -1;
      for (int j = begin; j < end; ++j) {
        int tree = flex_tree_ids[j];
        if (tree < 0 || tree >= ntree) { sleep_status[world] = 1; break; }
        if (tree_state[state_base + tree] < 0) { awake_tree = tree; break; }
      }
      if (sleep_status[world] != 0 || awake_tree < 0) continue;
      int wake_value = tree_state[state_base + awake_tree];
      for (int j = begin; j < end; ++j) {
        int tree = flex_tree_ids[j];
        if (tree_state[state_base + tree] >= 0) {
          if (!wake_sleep_cycle(tree_state, state_base, ntree, tree, wake_value))
            sleep_status[world] = 1;
          break;
        }
      }
    }
  }

  // Wake checks run in the current position stage before smooth forces and
  // constraints. Sleep countdown belongs to mj_advance after integration.
  if (advance_mode[0] == 0 && sleep_status[world] == 0) {
    for (int tree = 0; tree < ntree; ++tree) {
      tree_awake[state_base + tree] = tree_state[state_base + tree] < 0 ? 1 : 0;
    }
    return;
  }

  if (sleep_status[world] == 0) {
    // Advance each awake tree's countdown. Trees receiving an applied force
    // were set to the fully-awake sentinel above.
    for (int tree = 0; tree < ntree; ++tree) {
      int state = tree_state[state_base + tree];
      bool can_sleep = tree_policy[tree] != 1 && tree_policy[tree] != 3;
      int dof_adr = tree_dofadr[tree];
      int dof_count = tree_dofnum[tree];
      float weighted_speed = 0.0f;
      for (int j = 0; j < dof_count; ++j) {
        int dof = dof_adr + j;
        weighted_speed = max(weighted_speed,
            fabs(qvel[qvel_base + dof]) * dof_length[dof]);
        can_sleep = can_sleep && qfrc_applied[qvel_base + dof] == 0.0f;
      }
      can_sleep = can_sleep && (tolerance > 0.0f
          ? weighted_speed < tolerance : weighted_speed == 0.0f);
      int body_adr = tree_bodyadr[tree];
      int body_count = tree_bodynum[tree];
      for (int j = 0; j < body_count; ++j) {
        int body = body_adr + j;
        for (int axis = 0; axis < 6; ++axis) {
          can_sleep = can_sleep && xfrc_applied[body_base + body * 6 + axis] == 0.0f;
        }
      }
      if (!can_sleep) {
        tree_state[state_base + tree] = -11;
      } else if (state < -1) {
        tree_state[state_base + tree] = state + 1;
      }
      tree_awake[state_base + tree] = tree_state[state_base + tree] < 0 ? 1 : 0;
    }

    // Candidate trees at -1 sleep together only after every tree connected
    // by an active row is ready. tree_awake is temporary component-label
    // storage here, then becomes the final awake mask below.
    for (int tree = 0; tree < ntree; ++tree) {
      tree_awake[state_base + tree] = tree_state[state_base + tree] == -1
          ? tree + 1 : 0;
    }
    for (int pass = 0; pass < ntree; ++pass) {
      bool changed = false;
      for (int link = 0; link < max_links; ++link) {
        int a = tree_links[(int(world) * max_links + link) * 2 + 0];
        int b = tree_links[(int(world) * max_links + link) * 2 + 1];
        if (a < 0 || b < 0 || a >= ntree || b >= ntree || a == b) continue;
        int la = tree_awake[state_base + a];
        int lb = tree_awake[state_base + b];
        if (la > 0 && lb > 0) {
          int label = min(la, lb);
          if (la != label) { tree_awake[state_base + a] = label; changed = true; }
          if (lb != label) { tree_awake[state_base + b] = label; changed = true; }
        } else if (la > 0 && tree_state[state_base + b] < -1) {
          tree_awake[state_base + a] = 0;
          changed = true;
        } else if (lb > 0 && tree_state[state_base + a] < -1) {
          tree_awake[state_base + b] = 0;
          changed = true;
        }
      }
      for (int eq = 0; eq < neq; ++eq) {
        if (equality_active[world * neq + eq] == 0) continue;
        int kind = equality_kind[eq];
        if (kind == 1) {
          int a = equality_pair_trees[eq * 2 + 0];
          int b = equality_pair_trees[eq * 2 + 1];
          if (a < 0 || b < 0 || a >= ntree || b >= ntree || a == b) continue;
          int la = tree_awake[state_base + a];
          int lb = tree_awake[state_base + b];
          if (la > 0 && lb > 0) {
            int label = min(la, lb);
            if (la != label) { tree_awake[state_base + a] = label; changed = true; }
            if (lb != label) { tree_awake[state_base + b] = label; changed = true; }
          } else if (la > 0 && tree_state[state_base + b] < -1) {
            tree_awake[state_base + a] = 0;
            changed = true;
          } else if (lb > 0 && tree_state[state_base + a] < -1) {
            tree_awake[state_base + b] = 0;
            changed = true;
          }
        } else if (kind == 2) {
          int begin = flex_offsets[eq];
          int end = flex_offsets[eq + 1];
          int root = -1;
          for (int j = begin; j < end; ++j) {
            int tree = flex_tree_ids[j];
            if (tree >= 0 && tree < ntree) { root = tree; break; }
          }
          if (root >= 0) {
            for (int j = begin + 1; j < end; ++j) {
              int a = root, b = flex_tree_ids[j];
              if (b < 0 || b >= ntree || a == b) continue;
              int la = tree_awake[state_base + a];
              int lb = tree_awake[state_base + b];
              if (la > 0 && lb > 0) {
                int label = min(la, lb);
                if (la != label) { tree_awake[state_base + a] = label; changed = true; }
                if (lb != label) { tree_awake[state_base + b] = label; changed = true; }
              } else if (la > 0 && tree_state[state_base + b] < -1) {
                tree_awake[state_base + a] = 0;
                changed = true;
              } else if (lb > 0 && tree_state[state_base + a] < -1) {
                tree_awake[state_base + b] = 0;
                changed = true;
              }
            }
          }
        }
      }
      if (!changed) break;
    }
    for (int tree = 0; tree < ntree; ++tree) {
      int label = tree_awake[state_base + tree];
      if (label <= 0) continue;
      int next = -1;
      for (int candidate = 0; candidate < ntree; ++candidate) {
        if (tree_awake[state_base + candidate] == label && candidate > tree
            && (next < 0 || candidate < next)) next = candidate;
      }
      if (next < 0) {
        for (int candidate = 0; candidate <= tree; ++candidate) {
          if (tree_awake[state_base + candidate] == label
              && (next < 0 || candidate < next)) next = candidate;
        }
      }
      tree_state[state_base + tree] = next;
    }
  }

  if (sleep_status[world] == 3) {
    for (int tree = 0; tree < ntree; ++tree) {
      tree_awake[state_base + tree] = tree_state[state_base + tree] < 0 ? 1 : 0;
    }
    return;
  }
  if (sleep_status[world] != 0) {
    for (int tree = 0; tree < ntree; ++tree) {
      tree_state[state_base + tree] = -11;
      tree_awake[state_base + tree] = 1;
    }
    return;
  }
  for (int tree = 0; tree < ntree; ++tree) {
    tree_awake[state_base + tree] = tree_state[state_base + tree] < 0 ? 1 : 0;
  }
}

// Build the fixed-width MuJoCo 3.10 awake lists consumed by smooth dynamics
// and RNE. Static bodies participate in the body/parent passes; only dynamic
// awake trees contribute generalized DOFs. Source order is retained exactly.
kernel void build_awake_lists(
    device const int* tree_awake [[buffer(0)]],
    device const int* body_treeid [[buffer(1)]],
    device const int* body_parentid [[buffer(2)]],
    device const int* dof_treeid [[buffer(3)]],
    device int* body_ids [[buffer(4)]],
    device int* parent_ids [[buffer(5)]],
    device int* dof_ids [[buffer(6)]],
    device int* body_mask [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant int* dims [[buffer(9)]],
    device const int* active_worlds [[buffer(10)]],
    uint world [[thread_position_in_grid]]) {
  int nbody = dims[0], nv = dims[1], ntree = dims[2], batch = dims[3];
  if (int(world) >= batch) return;
  if (active_worlds[world] == 0) return;
  int body_stride = max(nbody, 1);
  int dof_stride = max(nv, 1);
  int tree_stride = max(ntree, 1);
  int body_base = int(world) * body_stride;
  int dof_base = int(world) * dof_stride;
  int tree_base = int(world) * tree_stride;
  for (int i = 0; i < body_stride; ++i) {
    body_ids[body_base + i] = -1;
    parent_ids[body_base + i] = -1;
    body_mask[body_base + i] = 0;
  }
  for (int i = 0; i < dof_stride; ++i) dof_ids[dof_base + i] = -1;
  int nbody_awake = 0;
  int nparent_awake = 0;
  int nv_awake = 0;
  for (int body = 0; body < nbody; ++body) {
    int tree = body_treeid[body];
    bool awake = tree < 0 || (tree < ntree && tree_awake[tree_base + tree] != 0);
    if (awake) {
      body_ids[body_base + nbody_awake++] = body;
      body_mask[body_base + body] = 1;
    }
    if (body > 0) {
      int parent = body_parentid[body];
      int parent_tree = body_treeid[parent];
      bool parent_awake = parent_tree < 0
          || (parent_tree < ntree && tree_awake[tree_base + parent_tree] != 0);
      if (parent_awake) parent_ids[body_base + nparent_awake++] = body;
    }
  }
  for (int dof = 0; dof < nv; ++dof) {
    int tree = dof_treeid[dof];
    if (tree >= 0 && tree < ntree && tree_awake[tree_base + tree] != 0)
      dof_ids[dof_base + nv_awake++] = dof;
  }
  counts[int(world) * 3 + 0] = nbody_awake;
  counts[int(world) * 3 + 1] = nparent_awake;
  counts[int(world) * 3 + 2] = nv_awake;
}

kernel void reset_sleep_rows(
    device const int* selected [[buffer(0)]],
    device const int* initial_tree_state [[buffer(1)]],
    device const int* eq_active_default [[buffer(2)]],
    device int* tree_state [[buffer(3)]],
    device int* tree_awake [[buffer(4)]],
    device int* links [[buffer(5)]],
    device int* constraints_active [[buffer(6)]],
    device int* links_overflow [[buffer(7)]],
    device int* status [[buffer(8)]],
    device int* eq_active [[buffer(9)]],
    device float* zero_qfrc [[buffer(10)]],
    device float* zero_xfrc [[buffer(11)]],
    constant int* dims [[buffer(12)]],
    uint world [[thread_position_in_grid]]) {
  int batch=dims[0], ntree=dims[1], max_links=dims[2], neq=dims[3];
  int nv=dims[4], nbody=dims[5];
  if (world>=uint(batch) || selected[world]==0) return;
  uint tb=world*uint(max(ntree,1));
  for (int t=0;t<ntree;++t) {
    tree_state[tb+uint(t)]=initial_tree_state[t];
    tree_awake[tb+uint(t)]=initial_tree_state[t]<0?1:0;
  }
  int lb=int(world)*max(max_links,1)*2;
  for (int i=0;i<2*max_links;++i) links[lb+i]=-1;
  int eb=int(world)*neq;
  for (int i=0;i<neq;++i) {
    constraints_active[eb+i]=0;
    eq_active[eb+i]=eq_active_default[eb+i];
  }
  links_overflow[world]=0;
  status[world]=0;
  uint vb=world*uint(max(nv,1));
  for (int i=0;i<nv;++i) zero_qfrc[vb+uint(i)]=0.0f;
  uint xb=world*uint(nbody)*6u;
  for (int i=0;i<nbody*6;++i) zero_xfrc[xb+uint(i)]=0.0f;
}

// Convert a compact MPS bool mask into the scheduler's reusable int32 ABI.
kernel void copy_active_worlds_bool(
    device const uchar* active_worlds [[buffer(0)]],
    device int* active_worlds_int [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint world [[thread_position_in_grid]]) {
  int batch = dims[3];
  if (int(world) >= batch) return;
  active_worlds_int[world] = active_worlds[world] != 0 ? 1 : 0;
}

// mj_sleepTrees clears qacc as well as qvel when a tree enters its sleep
// cycle. Run after the pre-integration transition, before saving qacc or its
// warmstart copy. This writes only inactive dynamic-tree coordinates.
kernel void zero_asleep_acceleration(
    device float* qacc [[buffer(0)]],
    device const int* dof_treeid [[buffer(1)]],
    device const int* tree_awake [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], ntree=dims[1], batch=dims[2];
  if (int(world)>=batch) return;
  for (int dof=0; dof<nv; ++dof) {
    int tree=dof_treeid[dof];
    if (tree>=0 && tree<ntree
        && tree_awake[int(world)*max(ntree,1)+tree]==0)
      qacc[int(world)*nv+dof]=0.0f;
  }
}
