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
    uint world [[thread_position_in_grid]]) {
  int nv = dims[0];
  int nbody = dims[1];
  int ntree = dims[3];
  int max_links = dims[4];
  int batch = dims[5];
  bool enabled = dims[6] != 0;
  bool island_disabled = dims[7] != 0;
  if (int(world) >= batch) return;
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

  float tolerance = sleep_tolerance[0];
  for (int tree = 0; tree < ntree; ++tree) {
    int state = tree_state[state_base + tree];
    bool can_sleep = tree_policy[tree] != 1 && tree_policy[tree] != 3;
    bool perturbed = !can_sleep;
    int dof_adr = tree_dofadr[tree];
    int dof_count = tree_dofnum[tree];
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
    if ((state >= 0 && perturbed) || (state < 0 && !can_sleep)) {
      if (state >= 0 && !wake_sleep_cycle(tree_state, state_base, ntree, tree, -11)) {
        sleep_status[world] = 1;
      } else if (state < 0) {
        tree_state[state_base + tree] = -11;
      }
    }
  }

  // Existing sleeping cycles wake as a unit if their eligibility was lost.
  if (sleep_status[world] == 0) {
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

  // Contacts/equalities wake whole previously associated cycles. Two
  // independently sleeping cycles joined by a new active row both wake.
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
      } else if (sa >= 0 && sb >= 0
                 && !same_sleep_cycle(tree_state, state_base, ntree, a, b)) {
        if (!wake_sleep_cycle(tree_state, state_base, ntree, a, -11)
            || !wake_sleep_cycle(tree_state, state_base, ntree, b, -11)) {
          sleep_status[world] = 1;
          break;
        }
        changed = true;
      }
    }
    if (!changed) break;
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
