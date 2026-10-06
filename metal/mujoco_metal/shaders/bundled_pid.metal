// MuJoCo 3.10.0 bundled mujoco.pid actuator plugin.
#include <metal_stdlib>
using namespace metal;

kernel void bundled_pid_stage(
    device const float* control [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* activation_dot [[buffer(2)]],
    device const float* length [[buffer(3)]],
    device const float* velocity [[buffer(4)]],
    device const int* plugin_mask [[buffer(5)]],
    device const float* params [[buffer(6)]],
    device const int* flags [[buffer(7)]],
    device const int* actadr [[buffer(8)]],
    device const int* actnum [[buffer(9)]],
    device const int* dyntype [[buffer(10)]],
    device const int* actearly [[buffer(11)]],
    device const int* actlimited [[buffer(12)]],
    device const float* actrange [[buffer(13)]],
    device const float* dynprm [[buffer(14)]],
    device const float* time [[buffer(15)]],
    constant int* dims [[buffer(16)]],
    constant float* dt_data [[buffer(17)]],
    device float* plugin_force [[buffer(18)]],
    device const int* ctrllimited [[buffer(19)]],
    device const float* ctrlrange [[buffer(20)]],
    uint world [[thread_position_in_grid]]) {
  int nu = dims[0], na = dims[1], batch = dims[2];
  int actuation_disabled = dims[3], clamp_disabled = dims[4];
  if (world >= uint(batch)) return;
    float h = dt_data[0];
  uint ubase = world * uint(max(nu, 1));
  uint abase = world * uint(max(na, 1));
  for (int a = 0; a < nu; ++a) {
    plugin_force[ubase + uint(a)] = 0.0f;
    if (actuation_disabled || !plugin_mask[a]) continue;
    int first = actadr[a], n = actnum[a];
    // P-only and PD-only PID configurations are stateless and are compiled
    // with actadr=-1, actnum=0. They still compute force from control/length.
    if ((n > 0 && (first < 0 || first + n > na)) ||
        (n == 0 && (dyntype[a] != 0 || flags[2 * uint(a)] ||
                    flags[2 * uint(a) + 1]))) continue;
    float ctrl_actdot = control[ubase + uint(a)];
    float ctrl_force = ctrl_actdot;
    // The upstream plugin's GetCtrl clips d->ctrl itself for DYN_NONE, even
    // when mjDSBL_CLAMPCTRL is set for the generic actuator path.
    if (dyntype[a] == 0 && ctrllimited[a]) {
      ctrl_actdot = clamp(ctrl_actdot, ctrlrange[2 * uint(a)],
                          ctrlrange[2 * uint(a) + 1]);
      ctrl_force = ctrl_actdot;
    }
    if (dyntype[a] != 0) {
      int last = first + n - 1;
      ctrl_actdot = activation[abase + uint(last)];
      ctrl_force = ctrl_actdot;
      if (actearly[a]) {
        float dot = activation_dot[abase + uint(last)];
        float value = ctrl_actdot + dot * h;
        if (dyntype[a] == 3) {
          float tau = max(1.0e-15f, dynprm[10 * uint(a)]);
          value = ctrl_actdot + dot * tau * (1.0f - exp(-h / tau));
        }
        if (actlimited[a]) {
          value = clamp(value, actrange[2 * uint(a)], actrange[2 * uint(a) + 1]);
        }
        ctrl_force = value;
      }
    }

    device const float* p = params + 5 * uint(a);
    device const int* f = flags + 2 * uint(a);
    int slot = first;
    float integral_state = 0.0f;
    if (f[0]) integral_state = activation[abase + uint(slot++)];
    float previous_ctrl = 0.0f;
    if (f[1]) previous_ctrl = activation[abase + uint(slot)];
    // MuJoCo's previous-control state is considered initialized when d->time
    // is positive. The caller supplies each world's current time tensor.
    if (f[1] && time[world] > 0.0f) {
      float delta = p[4] * h;
      ctrl_actdot = clamp(ctrl_actdot, previous_ctrl - delta, previous_ctrl + delta);
      ctrl_force = clamp(ctrl_force, previous_ctrl - delta, previous_ctrl + delta);
    }
    float error_actdot = ctrl_actdot - length[ubase + uint(a)];
    float error_force = ctrl_force - length[ubase + uint(a)];
    int last = first + n - 1;
    if (f[0]) {
      float next_integral = integral_state + error_actdot * h;
      if (p[3] >= 0.0f) next_integral = clamp(next_integral, -p[3], p[3]);
      activation_dot[abase + uint(first)] = (next_integral - integral_state) / h;
    }
    if (f[1]) {
      int slew_slot = first + (f[0] ? 1 : 0);
      activation_dot[abase + uint(slew_slot)] = (ctrl_actdot - previous_ctrl) / h;
    }
    float ctrl_dot = dyntype[a] == 0 ? 0.0f : activation_dot[abase + uint(last)];
    float force_integral = 0.0f;
    if (f[0]) {
      force_integral = integral_state + error_force * h;
      if (p[3] >= 0.0f) force_integral = clamp(force_integral, -p[3], p[3]);
    }
    plugin_force[ubase + uint(a)] =
        p[0] * error_force + p[2] * (ctrl_dot - velocity[ubase + uint(a)]) +
        p[1] * force_integral;
  }
}
