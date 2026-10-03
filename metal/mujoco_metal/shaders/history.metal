// Copyright 2021 DeepMind Technologies Limited
// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Ports the canonical ring operations in engine_util_misc.c and sensor
// compute/read and mj_advance ordering in engine_sensor.c/engine_forward.c.
#include <metal_stdlib>
using namespace metal;

inline int history_index(int cursor, int n, int logical) {
  return (cursor+1+logical)%n;
}

inline int history_find(device const float* times, int n, int cursor, float t) {
  if (t<=times[history_index(cursor,n,0)]) return 0;
  if (t>times[cursor]) return n;
  int lo=0,hi=n-1;
  while(hi-lo>1) {
    int mid=(lo+hi)/2;
    if(times[history_index(cursor,n,mid)]<t) lo=mid; else hi=mid;
  }
  return hi;
}

inline float history_read(device const float* buf, int n, int dim, int channel,
                          float2 query, int interp) {
  float t=query.x;
  int cursor=int(buf[1]);
  device const float* times=buf+2;
  device const float* values=times+n;
  int oldest=history_index(cursor,n,0);
  if(t-times[oldest]+query.y<=1e-15f) return values[oldest*dim+channel];
  if(t-times[cursor]+query.y>=-1e-15f) return values[cursor*dim+channel];
  int i=history_find(times,n,cursor,t),hi=history_index(cursor,n,i);
  if(i>0 && t==times[hi] && query.y>1e-15f) {i++;hi=history_index(cursor,n,i);}

  if(abs(t-times[hi]+query.y)<1e-15f) return values[hi*dim+channel];
  int lo=history_index(cursor,n,i-1);
  if(interp==0) return values[lo*dim+channel];
  float dt=times[hi]-times[lo],alpha=(t-times[lo]+query.y)/dt;
  float vlo=values[lo*dim+channel],vhi=values[hi*dim+channel];
  if(interp==1) return vlo+alpha*(vhi-vlo);
  float ml=0,mh=0;
  if(i>1) {
    int prev=history_index(cursor,n,i-2);
    ml=(vhi-values[prev*dim+channel])/(times[hi]-times[prev]);
  }
  if(i<n-1) {
    int next=history_index(cursor,n,i+1);
    mh=(values[next*dim+channel]-vlo)/(times[next]-times[lo]);
  }
  float a2=alpha*alpha,a3=a2*alpha;
  return (2*a3-3*a2+1)*vlo+(a3-2*a2+alpha)*dt*ml
      +(-2*a3+3*a2)*vhi+(a3-a2)*dt*mh;
}

inline int history_insert(device float* buf, int n, int dim, float t) {
  int cursor=int(buf[1]);
  device float* times=buf+2;
  device float* values=times+n;
  int i=history_find(times,n,cursor,t);
  if(i<n) {
    int p=history_index(cursor,n,i);
    if(abs(t-times[p])<1e-15f) return p;
  }
  if(i==0) {int p=history_index(cursor,n,0);times[p]=t;return p;}
  if(i==n) {cursor=(cursor+1)%n;buf[1]=cursor;times[cursor]=t;return cursor;}
  for(int j=0;j<i-1;j++) {
    int src=history_index(cursor,n,j+1),dst=history_index(cursor,n,j);
    times[dst]=times[src];
    for(int c=0;c<dim;c++) values[dst*dim+c]=values[src*dim+c];
  }
  int p=history_index(cursor,n,i-1);times[p]=t;return p;
}

// Recover the low part of a compiled interval clock from its tick lattice.
// Canonical history retains the rounded user value, including after restore.
// Volatile intermediates prevent fast-math reassociation from erasing the
// rounding residuals. No double arithmetic or host timing decisions are used.
inline float2 clock_add(float2 a, float2 b) {
  volatile float sum=a.x+b.x;
  volatile float v=sum-a.x;
  volatile float av=sum-v;
  volatile float da=a.x-av;
  volatile float db=b.x-v;
  volatile float error=(da+db)+(a.y+b.y);
  volatile float hi=sum+error;
  volatile float lo=error-(hi-sum);
  return float2(hi,lo);
}
inline float2 clock_tick(float tick, device const float* p) {
  volatile float product=tick*p[1];
  volatile float error=fma(tick,p[1],-product)+tick*p[2];
  return clock_add(float2(product,error),float2(p[3],p[4]));
}
inline float2 history_next_tick(float previous, device const float* p) {
  float tick=round((previous-p[3])/p[1]);
  float2 current=clock_tick(tick,p);
  if(previous==current.x) return clock_tick(tick+1,p);
  return clock_add(float2(previous,0),float2(p[1],p[2]));
}
inline bool history_tick_due(float previous, float time, device const float* p) {
  float2 next=history_next_tick(previous,p);
  return next.x<time || (next.x==time && next.y<=0);
}

kernel void history_samples(
    device const float* history [[buffer(0)]], device const float* time [[buffer(1)]],
    device float* samples [[buffer(2)]], device const int* meta [[buffer(3)]],
    device const float* parameters [[buffer(4)]], device const int* dims [[buffer(5)]],
    device const int* flags [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  int ne=dims[1];if(!ne||tid>=uint(dims[0]*ne))return;
  int w=int(tid)/ne,e=int(tid)%ne,kind=meta[6*e];
  if(kind!=flags[0])return;
  int adr=meta[6*e+1],n=meta[6*e+2],dim=meta[6*e+3],ip=meta[6*e+4],out=meta[6*e+5];
  device const float* buf=history+w*dims[2]+adr;
  float delay=parameters[6*e],period=parameters[6*e+1];
  bool read=flags[1]||delay>0||(period>0&&!history_tick_due(buf[0],time[w],parameters+6*e));
  if(read) {
    int stride=kind==0?dims[3]:dims[4];
    for(int c=0;c<dim;c++)samples[w*stride+out+c]=history_read(buf,n,dim,c,clock_add(float2(time[w],0),float2(-delay,-parameters[6*e+5])),ip);
  }
}

kernel void history_record(
    device float* history [[buffer(0)]], device const float* time [[buffer(1)]],
    device const float* controls [[buffer(2)]], device const float* sensors [[buffer(3)]],
    device const int* success [[buffer(4)]], device const int* meta [[buffer(5)]],
    device const float* parameters [[buffer(6)]], device const int* dims [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int ne=dims[1];if(!ne||tid>=uint(dims[0]*ne))return;
  int w=int(tid)/ne,e=int(tid)%ne;if(!success[w])return;
  int kind=meta[6*e],adr=meta[6*e+1],n=meta[6*e+2],dim=meta[6*e+3],out=meta[6*e+5];
  device float* buf=history+w*dims[2]+adr;
  float period=parameters[6*e+1];
  if(kind==1&&period>0) {
    if(!history_tick_due(buf[0],time[w],parameters+6*e))return;
    float2 next=history_next_tick(buf[0],parameters+6*e);
    buf[0]=next.x+next.y;
  }
  int p=history_insert(buf,n,dim,time[w]);
  device const float* raw=kind==0?controls+w*dims[3]+out:sensors+w*dims[4]+out;
  for(int c=0;c<dim;c++)buf[2+n+p*dim+c]=raw[c];
}
