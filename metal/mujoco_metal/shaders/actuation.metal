// Copyright 2026 The MuJoCo Metal contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

kernel void scalar_motor_force(
    device const float* ctrl [[buffer(0)]],
    device const int* dof [[buffer(1)]],
    device const float* gear [[buffer(2)]],
    device const float* gain [[buffer(3)]],
    device const int* ctrl_limited [[buffer(4)]],
    device const float* ctrl_range [[buffer(5)]],
    device const int* force_limited [[buffer(6)]],
    device const float* force_range [[buffer(7)]],
    device const int* actuator_group [[buffer(8)]],
    constant int* dims [[buffer(9)]],
    device float* qfrc [[buffer(10)]],
    device float* act_force [[buffer(11)]],
    uint world [[thread_position_in_grid]]) {
  int nu = dims[0];
  int nv = dims[1];
  int batch = dims[2];
  int actuation_disabled = dims[3];
  int clampctrl_disabled = dims[4];
  int disableactuator = dims[5];
  if (world >= uint(batch)) return;

  uint force_offset = world * uint(nv);
  uint act_offset = world * uint(nu);
  for (int v = 0; v < nv; ++v) qfrc[force_offset + uint(v)] = 0.0f;
  for (int a = 0; a < nu; ++a) act_force[act_offset + uint(a)] = 0.0f;
  if (actuation_disabled != 0) return;

  for (int actuator = 0; actuator < nu; ++actuator) {
    // Inspect exponent bits so fast-math cannot fold away the validity check.
    uint control_bits = as_type<uint>(ctrl[world * uint(nu) + uint(actuator)]);
    if ((control_bits & 0x7f800000u) == 0x7f800000u) {
      float invalid = as_type<float>(0x7fc00000u);
      for (int v = 0; v < nv; ++v) qfrc[force_offset + uint(v)] = invalid;
      for (int a = 0; a < nu; ++a) act_force[act_offset + uint(a)] = invalid;
      return;
    }
  }

  for (int actuator = 0; actuator < nu; ++actuator) {
    int group = actuator_group[actuator];
    if ((disableactuator & (1 << group)) != 0) continue;
    float control = ctrl[world * uint(nu) + uint(actuator)];
    if (clampctrl_disabled == 0 && ctrl_limited[actuator] != 0) {
      float lower = ctrl_range[2 * uint(actuator)];
      float upper = ctrl_range[2 * uint(actuator) + 1];
      control = clamp(control, lower, upper);
    }
    float force = gain[actuator] * control;
    if (force_limited[actuator] != 0) {
      float lower = force_range[2 * uint(actuator)];
      float upper = force_range[2 * uint(actuator) + 1];
      force = clamp(force, lower, upper);
    }
    act_force[act_offset + uint(actuator)] = force;
    qfrc[force_offset + uint(dof[actuator])] += gear[actuator] * force;
  }
}

// Full-family actuator dynamics and force stage for MuJoCo 3.10.0 (007).
// Pinned sources: engine/engine_forward.c mj_fwdActuation (act_dot switch,
// gain/bias switch, tendon/force clamps, DC mechanical forces, moment
// projection, gravcomp routing, joint clamp) and engine/engine_util_misc.c
// (muscle/DC utilities). One thread per world; fused act_dot then force in
// pinned order. NaN inputs propagate like the scalar stage.

inline float ad_clip(float x, float lo, float hi) {
  return min(max(x, lo), hi);
}
inline float ad_sigmoid(float x) {
  float xc = clamp(x, 0.0f, 1.0f);
  return xc * xc * xc * (xc * (xc * 6.0f - 15.0f) + 10.0f);
}

inline float ad_muscle_len_gain(float L, float lmin, float lmax) {
  if (L < lmin || L > lmax) return 0.0f;
  float a = 0.5f*(lmin+1.0f), b = 0.5f*(1.0f+lmax);
  if (L <= a) { float x=(L-lmin)/max(1e-15f,a-lmin); return 0.5f*x*x; }
  if (L <= 1.0f) { float x=(1.0f-L)/max(1e-15f,1.0f-a); return 1.0f-0.5f*x*x; }
  if (L <= b) { float x=(L-1.0f)/max(1e-15f,b-1.0f); return 1.0f-0.5f*x*x; }
  float x=(lmax-L)/max(1e-15f,lmax-b); return 0.5f*x*x;
}

// Full-family actuator dynamics/force/qfrc kernels for MuJoCo 3.10.0 (007).
// Pinned sources: engine/engine_forward.c mj_fwdActuation (act_dot switch,
// gain/bias switch, tendon/force clamps, DC mechanical forces), and
// engine/engine_support.c mj_nextActivation (exact slot forms).

inline float ad7_clip(float x, float lo, float hi) {
  return min(max(x, lo), hi);
}
inline float ad7_sigmoid(float x) {
  float xc = clamp(x, 0.0f, 1.0f);
  return xc * xc * xc * (xc * (xc * 6.0f - 15.0f) + 10.0f);
}

inline float ad7_muscle_len_gain(float L, float lmin, float lmax) {
  if (L < lmin || L > lmax) return 0.0f;
  float a = 0.5f*(lmin+1.0f), b = 0.5f*(1.0f+lmax);
  if (L <= a) { float x=(L-lmin)/max(1e-15f,a-lmin); return 0.5f*x*x; }
  if (L <= 1.0f) { float x=(1.0f-L)/max(1e-15f,1.0f-a); return 1.0f-0.5f*x*x; }
  if (L <= b) { float x=(L-1.0f)/max(1e-15f,b-1.0f); return 1.0f-0.5f*x*x; }
  float x=(lmax-L)/max(1e-15f,lmax-b); return 0.5f*x*x;
}

// Shared next-activation for one slot (pinned mj_nextActivation, minus the
// actrange clamp which the caller applies for non-DC-motor actuators).
inline float ad7_next_act(int dyntype, int slot_off, int cur_off, int br_off,
    int int_off, float cur, float dot, float vel,
    device const float* dynprm, device const float* gainprm,
    device const float* biasprm, int a, float h) {
  if (dyntype==3) {
    float tau=max(1e-15f,dynprm[10*a]);
    return cur+dot*tau*(1.0f-exp(-h/tau));
  }
  if (dyntype==5) {
    if (slot_off==cur_off) {
      float te=max(1e-15f,dynprm[10*a]);
      return cur+dot*te*(1.0f-exp(-h/te));
    }
    if (slot_off==br_off) {
      float fc=biasprm[10*a+3], fs=biasprm[10*a+4], vs=biasprm[10*a+5];
      float ratio=vel/max(1e-15f,vs);
      float g=fc+(fs-fc)*exp(-ratio*ratio);
      float dd=-dynprm[10*a+5]*abs(vel)/max(1e-15f,g);
      float eah=exp(dd*h);
      float inth=abs(dd)>1e-15f?(eah-1.0f)/dd:h;
      return eah*cur+inth*vel;
    }
    if (slot_off==int_off) {
      float nxt=cur+dot*h;
      float imax=dynprm[10*a+8];
      if (imax>0.0f) nxt=ad7_clip(nxt,-imax,imax);
      return nxt;
    }
    return cur+dot*h;
  }
  return cur+dot*h;
}
inline void ad7_slot_offsets(device const float* dynprm, device const float* gainprm,
    int a, thread int* cur_off, thread int* br_off, thread int* int_off) {
  int kk=0;
  if (dynprm[10*a+7]>0.0f) kk++;
  if (gainprm[10*a+5]>0.0f) { *int_off=kk; kk++; } else *int_off=-1;
  if (dynprm[10*a+2]>0.0f) kk++;
  if (dynprm[10*a+5]>0.0f) { *br_off=kk; kk++; } else *br_off=-1;
  if (dynprm[10*a]>0.0f) *cur_off=kk; else *cur_off=-1;
}

// Pinned engine_derivative.c mjd_muscleGain_vel. This is the engine's
// analytical integrator derivative, including the saturated curve branches.
inline float ad_muscle_gain_velocity(float len, float vel,
    device const float* range, float acc0, device const float* prm) {
  float force=prm[2]<0.0f?prm[3]/max(1e-15f,acc0):prm[2];
  float L0=(range[1]-range[0])/max(1e-15f,prm[1]-prm[0]);
  float L=prm[0]+(len-range[0])/max(1e-15f,L0);
  float V=vel/max(1e-15f,L0*prm[6]),y=prm[8]-1.0f;
  float dFV=V<=-1.0f?0.0f:V<=0.0f?2.0f*V+2.0f:
      V<=y?(-2.0f*V+2.0f*y)/max(1e-15f,y):0.0f;
  return -force*ad7_muscle_len_gain(L,prm[4],prm[5])*dFV/max(1e-15f,L0*prm[6]);
}

// Exact pinned mjd_actuator_vel block J' diag(dF/dvelocity) J. Each thread
// owns a matrix coefficient, avoiding atomics, host evaluation, and nv caps.
// Inputs belong to the same forward stage as actuator_force/act_dot.
kernel void actuator_velocity_derivative(
    device const float* ctrl [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device const float* act_dot [[buffer(2)]],
    device const float* length [[buffer(3)]],
    device const float* velocity [[buffer(4)]],
    device const float* force [[buffer(5)]],
    device const float* moment [[buffer(6)]],
    device const int* gaintype [[buffer(7)]],
    device const float* gainprm [[buffer(8)]],
    device const int* biastype [[buffer(9)]],
    device const float* biasprm [[buffer(10)]],
    device const int* dyntype [[buffer(11)]],
    device const float* dynprm [[buffer(12)]],
    device const int* actadr [[buffer(13)]],
    device const int* actnum [[buffer(14)]],
    device const int* actearly [[buffer(15)]],
    device const int* actlimited [[buffer(16)]],
    device const float* actrange [[buffer(17)]],
    device const int* group [[buffer(18)]],
    device const int* forcelimited [[buffer(19)]],
    device const float* forcerange [[buffer(20)]],
    device const float* lengthrange [[buffer(21)]],
    device const float* acc0 [[buffer(22)]],
    constant int* dims [[buffer(23)]],
    constant float* step_dt [[buffer(24)]],
    device float* derivative [[buffer(25)]],
    device const int* pattern [[buffer(26)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0],nu=dims[1],na=dims[2],batch=dims[3];
  if(tid>=uint(batch*nv*nv))return;
  int world=int(tid)/(nv*nv),row=(int(tid)/nv)%nv,col=int(tid)%nv;
  if(!pattern[row*nv+col]){derivative[tid]=0.0f;return;}
  float result=0.0f;
  if(!dims[4])for(int i=0;i<nu;i++) {
    if((dims[5]&(1<<group[i]))!=0)continue;
    int u=world*max(nu,1)+i,base=world*max(na,1);
    if(forcelimited[i]&&(force[u]<=forcerange[2*i]||force[u]>=forcerange[2*i+1]))continue;
    float bias_vel=0.0f,gain_vel=0.0f;
    if(biastype[i]==1)bias_vel=biasprm[10*i+2];
    else if(biastype[i]==3&&dynprm[10*i]<=0.0f)
      bias_vel=-gainprm[10*i+1]*gainprm[10*i+1]/max(1e-15f,gainprm[10*i]);
    if(gaintype[i]==1)gain_vel=gainprm[10*i+2];
    else if(gaintype[i]==2)gain_vel=ad_muscle_gain_velocity(
        length[u],velocity[u],lengthrange+2*i,acc0[i],gainprm+10*i);
    else if(gaintype[i]==3) {
      int mode=int(gainprm[10*i+8]);
      float dVdw=mode==1?-gainprm[10*i+6]:mode==2?-gainprm[10*i+4]:0.0f;
      float te=dynprm[10*i],R=max(1e-15f,gainprm[10*i]),K=gainprm[10*i+1];
      if(te>0.0f)bias_vel+=K*(dVdw-K)*(1.0f-exp(-step_dt[0]/te))/R;
      else bias_vel+=K*dVdw/R;
      if(dynprm[10*i+6]>0.0f)bias_vel-=dynprm[10*i+6];
    }
    if(gain_vel!=0.0f) {
      float value=ctrl[u];
      if(dyntype[i]!=0) {
        int first=actadr[i],last=first+actnum[i]-1;
        value=act[base+last];
        if(actearly[i]) {
          int current=-1,bristle=-1,integral=-1;
          ad7_slot_offsets(dynprm,gainprm,i,&current,&bristle,&integral);
          value=ad7_next_act(dyntype[i],last-first,current,bristle,integral,
              value,act_dot[base+last],velocity[u],dynprm,gainprm,biasprm,i,step_dt[0]);
          if(actlimited[i]&&dyntype[i]!=5)
            value=clamp(value,actrange[2*i],actrange[2*i+1]);
        }
      }
      bias_vel+=gain_vel*value;
    }
    int m=(world*max(nu,1)+i)*max(nv,1);
    result+=moment[m+row]*bias_vel*moment[m+col];
  }
  derivative[tid]=result;
}

// Stage 1: clipped control copy + act_dot switch (pinned mj_fwdActuation).
kernel void actuator_act_dot(
    device const float* ctrl [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device const float* length [[buffer(2)]],
    device const float* velocity [[buffer(3)]],
    device const int* dyntype [[buffer(4)]],
    device const float* dynprm [[buffer(5)]],
    device const float* gainprm [[buffer(6)]],
    device const float* biasprm [[buffer(7)]],
    device const int* actadr [[buffer(8)]],
    device const int* actnum [[buffer(9)]],
    device const int* ctrllimited [[buffer(10)]],
    device const float* ctrlrange [[buffer(11)]],
    constant int* dims [[buffer(12)]],
    constant float* step_dt [[buffer(13)]],
    device float* out_actdot [[buffer(14)]],
    device float* out_ctrl [[buffer(15)]],
    uint world [[thread_position_in_grid]]) {
  int nu=dims[0], na=dims[1], batch=dims[2];
  int act_disabled=dims[3], clamp_disabled=dims[4];
  float h=step_dt[0];
  if (world >= uint(batch)) return;
  uint ubase=world*uint(max(nu,1)), abase=world*uint(max(na,1));
  for (int j=0;j<na;++j) out_actdot[abase+uint(j)]=0.0f;
  for (int i=0;i<nu;++i) out_ctrl[ubase+uint(i)]=ctrl[ubase+uint(i)];
  if (act_disabled) return;
  device float* u=out_ctrl+ubase;
  for (int i=0;i<nu;++i) {
    float c=ctrl[ubase+uint(i)];
    if (!clamp_disabled && ctrllimited[i]) c=clamp(c,ctrlrange[2*i],ctrlrange[2*i+1]);
    u[i]=c;
  }
  for (int i=0;i<nu;++i) {
    int first=actadr[i], n=actnum[i];
    if (n<=0) continue;
    int dt=dyntype[i];
    int last=first+n-1;
    float a_last = (last>=0&&last<na)?act[abase+uint(last)]:0.0f;
    if (dt==1) {
      out_actdot[abase+uint(last)]=u[i];
    } else if (dt==2 || dt==3) {
      float tau=max(1e-15f,dynprm[10*i]);
      out_actdot[abase+uint(last)]=(u[i]-a_last)/tau;
    } else if (dt==4) {
      float cc=ad7_clip(u[i],0.0f,1.0f), aa=ad7_clip(a_last,0.0f,1.0f);
      float ta=dynprm[10*i]*(0.5f+1.5f*aa);
      float td=dynprm[10*i+1]/(0.5f+1.5f*aa);
      float w=dynprm[10*i+2], dc=cc-a_last, tau;
      if (w<1e-15f) tau=dc>0.0f?ta:td;
      else tau=td+(ta-td)*ad7_sigmoid(dc/w+0.5f);
      out_actdot[abase+uint(last)]=dc/max(1e-15f,tau);
    } else if (dt==5) {
      int adr=first;
      float vel=velocity[ubase+uint(i)], len=length[ubase+uint(i)];
      float r=gainprm[10*i], k=gainprm[10*i+1], ki=gainprm[10*i+5], te=dynprm[10*i];
      bool has_slew=dynprm[10*i+7]>0.0f, has_int=ki>0.0f;
      bool has_temp=dynprm[10*i+2]>0.0f, has_br=dynprm[10*i+5]>0.0f, has_cur=te>0.0f;
      float uu=u[i], xI=0.0f;
      if (has_slew) {
        float up=act[abase+uint(adr)], sl=dynprm[10*i+7]*h;
        float ue=ad7_clip(uu,up-sl,up+sl);
        out_actdot[abase+uint(adr)]=(ue-up)/h;
        uu=ue; adr++;
      }
      if (has_int) {
        xI=act[abase+uint(adr)];
        int mode=int(gainprm[10*i+8]);
        float imax=dynprm[10*i+8], ad2=uu;
        if (mode==1) ad2=uu-len;
        if (imax>0.0f) {
          if (xI>=imax) ad2=min(ad2,0.0f); else if (xI<=-imax) ad2=max(ad2,0.0f);
        }
        out_actdot[abase+uint(adr)]=ad2; adr++;
      }
      float V;
      {
        int mode=int(gainprm[10*i+8]);
        float kp=gainprm[10*i+4], kd=gainprm[10*i+6], vmax=gainprm[10*i+7];
        if (mode>0) {
          if (mode==1) V=kp*(uu-len)+ki*xI-kd*vel;
          else V=kp*(uu-vel)+ki*(xI-len);
        } else V=uu;
        if (vmax>0.0f) V=ad7_clip(V,-vmax,vmax);
      }
      if (has_temp) {
        float c=dynprm[10*i+3], ta=dynprm[10*i+4];
        float alpha=gainprm[10*i+2], t0=gainprm[10*i+3];
        float temp=act[abase+uint(adr)];
        float rr=r*(1.0f+alpha*(temp+ta-t0));
        float cur = te>0.0f ? act[abase+uint(last)] : (V-k*vel)/rr;
        out_actdot[abase+uint(adr)]=(rr*cur*cur-temp/dynprm[10*i+2])/c;
        adr++;
      }
      if (has_br) {
        float z=act[abase+uint(adr)];
        float fc=biasprm[10*i+3], fs=biasprm[10*i+4], vs=biasprm[10*i+5];
        float ratio=vel/max(1e-15f,vs);
        float g=fc+(fs-fc)*exp(-ratio*ratio);
        float aa=-dynprm[10*i+5]*abs(vel)/max(1e-15f,g);
        out_actdot[abase+uint(adr)]=aa*z+vel;
        adr++;
      }
      if (has_cur) {
        float dimax=dynprm[10*i+1];
        float idot=(V/r-k/r*vel-act[abase+uint(last)])/te;
        if (dimax>0.0f) idot=ad7_clip(idot,-dimax,dimax);
        out_actdot[abase+uint(last)]=idot;
      }
      u[i]=uu;
    }
  }
  for (int i=0;i<nu;++i) out_ctrl[ubase+uint(i)]=u[i];
}

// Stage 2: gain/bias/force with tendon + force-range clamps and DC mechanics.
kernel void actuator_force(
    device const float* ctrl_used [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device const float* act_dot [[buffer(2)]],
    device const float* length [[buffer(3)]],
    device const float* velocity [[buffer(4)]],
    device const int* gaintype [[buffer(5)]],
    device const float* gainprm [[buffer(6)]],
    device const int* biastype [[buffer(7)]],
    device const float* biasprm [[buffer(8)]],
    device const int* actadr [[buffer(9)]],
    device const int* actnum [[buffer(10)]],
    device const int* actearly [[buffer(11)]],
    device const int* dyntype [[buffer(12)]],
    device const float* dynprm [[buffer(13)]],
    device const int* forcelimited [[buffer(14)]],
    device const float* forcerange [[buffer(15)]],
    device const int* trntype [[buffer(16)]],
    device const int* trnid [[buffer(17)]],
    device const int* tendon_limited [[buffer(18)]],
    device const float* tendon_range [[buffer(19)]],
    device const int* actuator_group [[buffer(20)]],
    device const float* muscle_lengthrange [[buffer(21)]],
    device const float* muscle_acc0 [[buffer(22)]],
    constant int* dims [[buffer(23)]],
    constant float* step_dt [[buffer(24)]],
    device float* out_force [[buffer(25)]],
    device float* out_ctrl [[buffer(26)]],
    uint world [[thread_position_in_grid]]) {
  int nu=dims[0], na=dims[1], batch=dims[2];
  int act_disabled=dims[3], disableactuator=dims[4], ntendon=dims[5];
  float h=step_dt[0];
  if (world >= uint(batch)) return;
  uint ubase=world*uint(max(nu,1)), abase=world*uint(max(na,1));
  for (int i=0;i<nu;++i) { out_force[ubase+uint(i)]=0.0f; out_ctrl[ubase+uint(i)]=ctrl_used[ubase+uint(i)]; }
  if (act_disabled) return;
  // Per-world owned workspace replaces fixed thread-local actuator arrays.
  // Serial source ordering still applies to tendon/force/DC clamps below.
  device float* u=out_ctrl+ubase;
  device float* f=out_force+ubase;
  for (int i=0;i<nu;++i) {
    int grp=actuator_group[i];
    if ((disableactuator&(1<<grp))!=0) continue;
    int n=actnum[i], first=actadr[i], last=first+n-1;
    float len=length[ubase+uint(i)], vel=velocity[ubase+uint(i)];
    float uu=u[i];
    float dp0=dynprm[10*i];
    int gt=gaintype[i];
    float g=0.0f;
    if (gt==0) g=gainprm[10*i];
    else if (gt==1) g=gainprm[10*i]+gainprm[10*i+1]*len+gainprm[10*i+2]*vel;
    else if (gt==2) {
      float rng0=gainprm[10*i], rng1=gainprm[10*i+1];
      float force=gainprm[10*i+2], scale=gainprm[10*i+3];
      float lmin=gainprm[10*i+4], lmax=gainprm[10*i+5], vmax=gainprm[10*i+6], fvmax=gainprm[10*i+8];
      float acc0=muscle_acc0[i];
      float lr0=muscle_lengthrange[2*i], lr1=muscle_lengthrange[2*i+1];
      if (force<0.0f) force=scale/max(1e-15f,acc0);
      float l0=(lr1-lr0)/max(1e-15f,rng1-rng0);
      float ln=rng0+(len-lr0)/max(1e-15f,l0);
      float vn=vel/max(1e-15f,l0*vmax);
      float fl=ad7_muscle_len_gain(ln,lmin,lmax);
      float y=fvmax-1.0f, fv;
      if (vn<=-1.0f) fv=0.0f;
      else if (vn<=0.0f) fv=(vn+1.0f)*(vn+1.0f);
      else if (vn<=y) fv=fvmax-(y-vn)*(y-vn)/max(1e-15f,y);
      else fv=fvmax;
      g=-force*fl*fv;
    } else {
      float r=gainprm[10*i], k=gainprm[10*i+1];
      float te=dp0;
      float rr=r;
      int cur_off=-1, br_off=-1, int_off=-1;
      ad7_slot_offsets(dynprm,gainprm,i,&cur_off,&br_off,&int_off);
      if (cur_off>=0 || br_off>=0 || int_off>=0 || true) {
        // temperature-adjusted resistance uses the temperature slot when present
        int kk=0;
        if (dynprm[10*i+7]>0.0f) kk++;
        if (gainprm[10*i+5]>0.0f) kk++;
        if (dynprm[10*i+2]>0.0f && n>0) {
          float temp=act[abase+uint(first+kk)];
          rr=r*(1.0f+gainprm[10*i+2]*(temp+dynprm[10*i+4]-gainprm[10*i+3]));
        }
      }
      g = te>0.0f ? k : k/max(1e-15f,rr);
      if (int(gainprm[10*i+8])>0) {
        float xI=0.0f;
        if (int_off>=0 && n>0) xI=act[abase+uint(first+int_off)];
        float kp=gainprm[10*i+4], ki=gainprm[10*i+5], kd=gainprm[10*i+6], vmax=gainprm[10*i+7];
        float V;
        if (int(gainprm[10*i+8])==1) V=kp*(uu-len)+ki*xI-kd*vel;
        else V=kp*(uu-vel)+ki*(xI-len);
        if (vmax>0.0f) V=ad7_clip(V,-vmax,vmax);
        uu=V;
      }
    }
    bool dcmotor_no_current = (gt==3 && dp0<=0.0f);
    float ff;
    if (n==0 || dcmotor_no_current) ff=g*uu;
    else {
      int adr=first+n-1;
      float aval;
      if (actearly[i]) {
        float dot=act_dot[abase+uint(adr)];
        int dt=dyntype[i];
        int cur_off=-1, br_off=-1, int_off=-1;
        ad7_slot_offsets(dynprm,gainprm,i,&cur_off,&br_off,&int_off);
        aval=ad7_next_act(dt,adr-first,cur_off,br_off,int_off,
          act[abase+uint(adr)],dot,vel,dynprm,gainprm,biasprm,i,h);
      } else aval=act[abase+uint(adr)];
      ff=g*aval;
    }
    float b=0.0f;
    int bt=biastype[i];
    if (bt==1) b=biasprm[10*i]+biasprm[10*i+1]*len+biasprm[10*i+2]*vel;
    else if (bt==2) {
      float rng0=biasprm[10*i], rng1=biasprm[10*i+1];
      float force=biasprm[10*i+2], scale=biasprm[10*i+3];
      float lmax=biasprm[10*i+5], fpmax=biasprm[10*i+7];
      float acc0=muscle_acc0[i];
      float lr0=muscle_lengthrange[2*i], lr1=muscle_lengthrange[2*i+1];
      if (force<0.0f) force=scale/max(1e-15f,acc0);
      float l0=(lr1-lr0)/max(1e-15f,rng1-rng0);
      float ln=rng0+(len-lr0)/max(1e-15f,l0);
      float bb=0.5f*(1.0f+lmax);
      if (ln>1.0f) {
        if (ln<=bb) { float x=(ln-1.0f)/max(1e-15f,bb-1.0f); b=-force*fpmax*0.5f*x*x; }
        else { float x=(ln-bb)/max(1e-15f,bb-1.0f); b=-force*fpmax*(0.5f+x); }
      }
    } else if (bt==3) {
      if (dp0<=0.0f) b-=g*gainprm[10*i+1]*vel;
    }
    f[i]=ff+b;
    u[i]=uu;
  }
  for (int i=0;i<nu;++i) out_ctrl[ubase+uint(i)]=u[i];
  for (int tt=0;tt<ntendon;++tt) {
    if (!tendon_limited[tt]) continue;
    float total=0.0f;
    for (int i=0;i<nu;++i) {
      if (trntype[i]!=3) continue;
      if (trnid[2*i]!=tt) continue;
      total+=f[i];
    }
    if (total!=0.0f) {
      float lo=tendon_range[2*tt], hi=tendon_range[2*tt+1];
      float s=1.0f;
      if (total<lo) s=lo/total; else if (total>hi) s=hi/total;
      if (s!=1.0f) for (int i=0;i<nu;++i) {
        if (trntype[i]!=3||trnid[2*i]!=tt) continue;
        f[i]*=s;
      }
    }
  }
  for (int i=0;i<nu;++i) {
    int grp=actuator_group[i];
    if ((disableactuator&(1<<grp))!=0) { f[i]=0.0f; continue; }
    if (forcelimited[i]) f[i]=clamp(f[i],forcerange[2*i],forcerange[2*i+1]);
  }
  for (int i=0;i<nu;++i) {
    int grp=actuator_group[i];
    if ((disableactuator&(1<<grp))!=0) continue;
    if (biastype[i]!=3) continue;
    float A=biasprm[10*i];
    if (A!=0.0f) f[i]+=A*sin(biasprm[10*i+1]*length[ubase+uint(i)]+biasprm[10*i+2]);
    float sigma0=dynprm[10*i+5];
    if (sigma0>0.0f && na>0) {
      int cur_off=-1, br_off=-1, int_off=-1;
      ad7_slot_offsets(dynprm,gainprm,i,&cur_off,&br_off,&int_off);
      if (br_off>=0) {
        int adr=actadr[i]+br_off;
        float z=(adr>=0&&adr<na)?act[abase+uint(adr)]:0.0f;
        float zdot=(adr>=0&&adr<na)?act_dot[abase+uint(adr)]:0.0f;
        f[i]-=sigma0*z+dynprm[10*i+6]*zdot;
      }
    }
  }
  for (int i=0;i<nu;++i) out_force[ubase+uint(i)]=f[i];
}

// Stage 3: qfrc assembly with gravcomp routing and joint clamps.
kernel void actuator_assemble_qfrc(
    device const float* force [[buffer(0)]],
    device const float* moment [[buffer(1)]],
    device const float* gravcomp [[buffer(2)]],
    device const int* jnt_limited [[buffer(3)]],
    device const float* jnt_range [[buffer(4)]],
    device const int* jnt_dofadr [[buffer(5)]],
    device const int* jnt_type [[buffer(6)]],
    device const int* jnt_gravcomp [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    device float* out_qfrc [[buffer(9)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nu=dims[1], batch=dims[2], njnt=dims[3];
  if (world >= uint(batch)) return;
  uint vbase=world*uint(max(nv,1)), ubase=world*uint(max(nu,1));
  for (int d=0;d<nv;++d) {
    float s=0.0f;
    for (int i=0;i<nu;++i) s+=moment[(ubase+uint(i))*uint(max(nv,1))+uint(d)]*force[ubase+uint(i)];
    out_qfrc[vbase+uint(d)]=s;
  }
  for (int j=0;j<njnt;++j) {
    if (jnt_gravcomp[j]==0) continue;
    int da=jnt_dofadr[j], ty=jnt_type[j];
    int nd=ty==0?6:(ty==1?3:1);
    for (int k=0;k<nd;++k) {
      int dof=da+k;
      if (dof>=0&&dof<nv) out_qfrc[vbase+uint(dof)]+=gravcomp[vbase+uint(dof)];
    }
  }
  for (int j=0;j<njnt;++j) {
    if (jnt_limited[j]==0) continue;
    int da=jnt_dofadr[j], ty=jnt_type[j];
    int nd=ty==0?6:(ty==1?3:1);
    float lo=jnt_range[2*j], hi=jnt_range[2*j+1];
    for (int k=0;k<nd;++k) {
      int dof=da+k;
      if (dof>=0&&dof<nv) out_qfrc[vbase+uint(dof)]=clamp(out_qfrc[vbase+uint(dof)],lo,hi);
    }
  }
}
// Activation advance for MuJoCo 3.10.0 (007).
// Pinned source: engine/engine_forward.c mj_step advance loop with
// engine/engine_support.c mj_nextActivation. FILTEREXACT and DC-motor
// current/bristle slots use exact integration; integral slots use Euler with
// anti-windup clamp; everything else uses Euler. Disabled actuators freeze
// (act_dot forced to zero); non-DC-motor slots clamp to actrange. The host
// skips this kernel entirely under mjDSBL_ACTUATION (pinned advance guard).
kernel void advance_activations(
    device const float* act [[buffer(0)]],
    device const float* act_dot [[buffer(1)]],
    device const float* velocity [[buffer(2)]],
    device const int* dyntype [[buffer(3)]],
    device const float* dynprm [[buffer(4)]],
    device const float* gainprm [[buffer(5)]],
    device const float* biasprm [[buffer(6)]],
    device const int* actadr [[buffer(7)]],
    device const int* actnum [[buffer(8)]],
    device const int* actlimited [[buffer(9)]],
    device const float* actrange [[buffer(10)]],
    device const int* actuator_group [[buffer(11)]],
    constant int* dims [[buffer(12)]],
    constant float* step_dt [[buffer(13)]],
    device float* out_act [[buffer(14)]],
    uint world [[thread_position_in_grid]]) {
  int nu=dims[0], na=dims[1], batch=dims[2];
  int disableactuator=dims[3];
  float h=step_dt[0];
  if (world >= uint(batch)) return;
  uint abase=world*uint(max(na,1)), ubase=world*uint(max(nu,1));
  for (int j=0;j<na;++j) out_act[abase+uint(j)]=act[abase+uint(j)];
  for (int i=0;i<nu;++i) {
    int first=actadr[i], n=actnum[i];
    if (n<=0) continue;
    int grp=actuator_group[i];
    bool dis=(disableactuator&(1<<grp))!=0;
    int dt=dyntype[i];
    for (int j=first;j<first+n;++j) {
      if (j<0||j>=na) continue;
      float dot=dis?0.0f:act_dot[abase+uint(j)];
      float cur=act[abase+uint(j)];
      float nxt=cur+dot*h;
      if (dt==3) {
        float tau=max(1e-15f,dynprm[10*i]);
        nxt=cur+dot*tau*(1.0f-exp(-h/tau));
      } else if (dt==5) {
        int off=j-first;
        int kk=0;
        if (dynprm[10*i+7]>0.0f) kk++;
        if (gainprm[10*i+5]>0.0f) kk++;
        int temp_off=-1, br_off=-1, cur_off=-1, int_off=-1;
        if (dynprm[10*i+2]>0.0f) { temp_off=kk; kk++; }
        if (dynprm[10*i+5]>0.0f) { br_off=kk; kk++; }
        if (dynprm[10*i]>0.0f) cur_off=kk;
        if (dynprm[10*i+7]>0.0f) { /* slew slot index 0 */ }
        // integral slot index: after slew if present
        int kk2=0;
        if (dynprm[10*i+7]>0.0f) kk2++;
        if (gainprm[10*i+5]>0.0f) { int_off=kk2; }
        if (off==cur_off) {
          float te=max(1e-15f,dynprm[10*i]);
          nxt=cur+dot*te*(1.0f-exp(-h/te));
        } else if (off==br_off) {
          float vel=velocity[ubase+uint(i)];
          float fc=biasprm[10*i+3], fs=biasprm[10*i+4], vs=biasprm[10*i+5];
          float ratio=vel/max(1e-15f,vs);
          float g=fc+(fs-fc)*exp(-ratio*ratio);
          float a=-dynprm[10*i+5]*abs(vel)/max(1e-15f,g);
          float eah=exp(a*h);
          float inth=abs(a)>1e-15f?(eah-1.0f)/a:h;
          nxt=eah*cur+inth*vel;
        } else if (off==int_off) {
          nxt=cur+dot*h;
          float imax=dynprm[10*i+8];
          if (imax>0.0f) nxt=ad_clip(nxt,-imax,imax);
        } else {
          nxt=cur+dot*h;
        }
      }
      out_act[abase+uint(j)]=nxt;
    }
    if (dt!=5 && actlimited[i]) {
      for (int j=first;j<first+n;++j) {
        if (j<0||j>=na) continue;
        out_act[abase+uint(j)]=clamp(out_act[abase+uint(j)],actrange[2*i],actrange[2*i+1]);
      }
    }
  }
}

// ---- Actuator control-history (delay line) ring ops (R06c) ----
// Pinned source: engine/engine_support.c mju_historyInsert (dim=1) and
// mju_historyRead plus engine/engine_forward.c mj_readCtrl. One thread per
// (world, actuator). Layout per actuator: times[nmax], values[nmax],
// cursor scalar; nsample/interp/delay from meta arrays. nsample==0 is a
// no-op (record) / live passthrough handled host-side (read).
// MINVAL matches mjMINVAL (1e-15).
inline int dl_phys(int cursor, int n, int logical) {
  return (cursor + 1 + logical) % n;
}
// Smallest logical i with times[phys(i)] >= t (linear scan == pinned
// circular binary search on sorted stamps, including stale zero slots).
inline int dl_find(device const float* times, int base, int cursor, int n, float t) {
  float t_oldest = times[base + dl_phys(cursor, n, 0)];
  float t_newest = times[base + dl_phys(cursor, n, n - 1)];
  if (t <= t_oldest) return 0;
  if (t > t_newest) return n;
  for (int i = 1; i < n; ++i) {
    if (times[base + dl_phys(cursor, n, i)] >= t) return i;
  }
  return n - 1;
}
kernel void delay_record(
    device const float* ctrl [[buffer(0)]],
    device const int* nsample [[buffer(1)]],
    device float* times [[buffer(2)]],
    device float* values [[buffer(3)]],
    device int* cursor [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    constant float* now [[buffer(6)]],
    device const int* record_mask [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int b = dims[0], nu = dims[1], nmax = dims[2];
  int world = int(tid) / max(nu, 1), i = int(tid) % max(nu, 1);
  if (world >= b || i >= nu) return;
  if (!record_mask[world * nu + i]) return;
  int n = nsample[i];
  if (n <= 0) return;
  float t = now[world];
  int base = (world * nu + i) * nmax;
  int cur = cursor[world * nu + i];
  int idx = dl_find(times + base, 0, cur, n, t);
  if (idx < n && abs(t - times[base + dl_phys(cur, n, idx)]) < 1e-15f) {
    values[base + dl_phys(cur, n, idx)] = ctrl[world * nu + i];
    return;
  }
  if (idx == 0) {
    int s = dl_phys(cur, n, 0);
    times[base + s] = t;
    values[base + s] = ctrl[world * nu + i];
    return;
  }
  if (idx == n) {
    cur = (cur + 1) % n;
    cursor[world * nu + i] = cur;
    times[base + cur] = t;
    values[base + cur] = ctrl[world * nu + i];
    return;
  }
  for (int j = 0; j < idx - 1; ++j) {
    int src = dl_phys(cur, n, j + 1), dst = dl_phys(cur, n, j);
    times[base + dst] = times[base + src];
    values[base + dst] = values[base + src];
  }
  int s = dl_phys(cur, n, idx - 1);
  times[base + s] = t;
  values[base + s] = ctrl[world * nu + i];
}
kernel void delay_read(
    device const float* times [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const int* cursor [[buffer(2)]],
    device const int* nsample [[buffer(3)]],
    device const int* interp [[buffer(4)]],
    device const float* qtime [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant int* dims [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int b = dims[0], nu = dims[1], nmax = dims[2];
  int world = int(tid) / max(nu, 1), i = int(tid) % max(nu, 1);
  if (world >= b || i >= nu) return;
  int n = nsample[i];
  if (n <= 0) { out[world * nu + i] = 0.0f; return; }
  // Query times are subtracted host-side in float64 (pinned mj_readCtrl
  // evaluates time-delay in mjtNum): float32 device subtraction loses
  // exact-stamp hits that the pinned eps comparison keeps.
  float t = qtime[world * nu + i];
  int base = (world * nu + i) * nmax;
  int cur = cursor[world * nu + i];
  int oldest = dl_phys(cur, n, 0), newest = dl_phys(cur, n, n - 1);
  if (t <= times[base + oldest] + 1e-15f) { out[world * nu + i] = values[base + oldest]; return; }
  if (t >= times[base + newest] - 1e-15f) { out[world * nu + i] = values[base + newest]; return; }
  int idx = dl_find(times + base, 0, cur, n, t);
  int hi = dl_phys(cur, n, idx);
  // Exact-hit tolerance is float32-adapted (1e-7): host query times are
  // differenced in float64 but stamps and delays quantize to float32,
  // shifting exact hits by ~1e-10 that pinned mjtNum arithmetic keeps.
  // Simulation steps (ms) stay far above this snapping radius.
  if (abs(t - times[base + hi]) < 1e-7f) { out[world * nu + i] = values[base + hi]; return; }
  int lo = dl_phys(cur, n, idx - 1);
  int ip = interp[i];
  if (ip == 0) { out[world * nu + i] = values[base + lo]; return; }
  float dt = times[base + hi] - times[base + lo];
  float alpha = (t - times[base + lo]) / dt;
  if (ip == 1) {
    out[world * nu + i] = values[base + lo] + alpha * (values[base + hi] - values[base + lo]);
    return;
  }
  float a2 = alpha * alpha, a3 = a2 * alpha;
  float h00 = 2.0f * a3 - 3.0f * a2 + 1.0f;
  float h10 = a3 - 2.0f * a2;
  float h01 = -2.0f * a3 + 3.0f * a2;
  float h11 = a3 - a2;
  float m_lo = 0.0f;
  if (idx > 1) {
    int p = dl_phys(cur, n, idx - 2);
    m_lo = (values[base + hi] - values[base + p]) / (times[base + hi] - times[base + p]);
  }
  float m_hi = 0.0f;
  if (idx < n - 1) {
    int q = dl_phys(cur, n, idx + 1);
    m_hi = (values[base + q] - values[base + lo]) / (times[base + q] - times[base + lo]);
  }
  out[world * nu + i] = h00 * values[base + lo] + h10 * dt * m_lo
      + h01 * values[base + hi] + h11 * dt * m_hi;
}
