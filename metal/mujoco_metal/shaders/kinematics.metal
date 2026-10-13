// Copy only selected rows of external pose inputs into the compact FK cache.
kernel void prepare_fk_rows(
    device const int* world_mask [[buffer(0)]],
    device const float* mocap_pos [[buffer(1)]],
    device const float* mocap_quat [[buffer(2)]],
    device const int* tree_awake [[buffer(3)]],
    device float* auxiliary [[buffer(4)]],
    constant uint* dims [[buffer(5)]],
    uint world [[thread_position_in_grid]]) {
  uint nbody=dims[1], batch=dims[5], nmocap=dims[6], ntree=dims[7];
  if (world >= batch || world_mask[world] == 0) return;
  uint mocap_values = nmocap ? batch*nmocap*7 : 1;
  uint mocap_id_offset = mocap_values+nbody;
  uint awake_offset = mocap_id_offset+nbody;
  uint valid_offset = awake_offset+batch*max(ntree, 1u);
  uint mismatch_offset = valid_offset+batch;
  if (nmocap) {
    uint pos_base=world*nmocap*3;
    uint quat_base=batch*nmocap*3+world*nmocap*4;
    for (uint i=0; i<nmocap*3; ++i)
      auxiliary[pos_base+i]=mocap_pos[pos_base+i];
    for (uint i=0; i<nmocap*4; ++i)
      auxiliary[quat_base+i]=mocap_quat[world*nmocap*4+i];
  }
  uint stride=max(ntree, 1u);
  for (uint i=0; i<stride; ++i) {
    auxiliary[awake_offset+world*stride+i] =
        float(tree_awake[world*stride+i]);
    auxiliary[mismatch_offset+world*stride+i]=0.0f;
  }
}

// One thread computes one world's complete pose tree. This favors a simple
// correctness baseline; parallel body traversal is a separate optimization.
kernel void forward_kinematics(
    device const int* static_int [[buffer(0)]],
    device const float* static_float [[buffer(1)]],
    device const float* qpos [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const int* world_mask [[buffer(4)]],
    device float* auxiliary [[buffer(5)]],
    constant uint* dims [[buffer(6)]],
    device const float* precision [[buffer(7)]],
    uint world [[thread_position_in_grid]]) {
  uint pb=world*2*dims[0];
  device const int* parent=static_int+dims[12];
  device const int* jnt_type=static_int+dims[13];
  device const int* jnt_qposadr=static_int+dims[14];
  device const int* jnt_bodyid=static_int+dims[15];
  device const int* body_sameframe=static_int+dims[16];
  device const int* geom_bodyid=static_int+dims[17];
  device const int* geom_sameframe=static_int+dims[18];
  device const int* site_bodyid=static_int+dims[19];
  device const int* site_sameframe=static_int+dims[20];
  device const float* body_pos=static_float+dims[21];
  device const float* body_quat=static_float+dims[22];
  device const float* jnt_pos=static_float+dims[23];
  device const float* jnt_axis=static_float+dims[24];
  device const float* qpos0=static_float+dims[25];
  device const float* geom_pos=static_float+dims[26];
  device const float* geom_quat=static_float+dims[27];
  device const float* site_pos=static_float+dims[28];
  device const float* site_quat=static_float+dims[29];
  device const float* body_ipos=static_float+dims[30];
  device const float* body_iquat=static_float+dims[31];
  device float* xpos=output+dims[32];
  device float* xpos_low=output+dims[33];
  device float* xpos_tail=output+dims[34];
  device float* xquat=output+dims[35];
  device float* xquat_low=output+dims[36];
  device float* xquat_tail=output+dims[37];
  device float* geom_xpos=output+dims[38];
  device float* geom_xpos_low=output+dims[39];
  device float* geom_xpos_tail=output+dims[40];
  device float* geom_xquat=output+dims[41];
  device float* geom_xquat_low=output+dims[42];
  device float* geom_xquat_tail=output+dims[43];
  device float* geom_xmat=output+dims[44];
  device float* geom_xmat_low=output+dims[45];
  device float* geom_xmat_tail=output+dims[46];
  device float* site_xpos=output+dims[47];
  device float* site_xpos_low=output+dims[48];
  device float* site_xpos_tail=output+dims[49];
  device float* site_xquat=output+dims[50];
  device float* site_xquat_low=output+dims[51];
  device float* site_xquat_tail=output+dims[52];
  device float* inertial_xpos=output+dims[53];
  device float* inertial_xpos_low=output+dims[54];
  device float* inertial_xpos_tail=output+dims[55];
  device float* inertial_xquat=output+dims[56];
  device float* inertial_xquat_low=output+dims[57];
  device float* inertial_xquat_tail=output+dims[58];
  device float* joint_anchor=output+dims[59];
  device float* joint_axis=output+dims[60];
  device float* joint_anchor_low=output+dims[61];
  device float* joint_anchor_tail=output+dims[62];
  device float* joint_axis_low=output+dims[63];
  device float* joint_axis_tail=output+dims[64];
  uint nq = dims[0], nbody = dims[1], njnt = dims[2];
  uint ngeom = dims[3], nsite = dims[4], batch = dims[5];
  uint nmocap = dims[6];
  uint ntree = dims[7];
  if (world >= batch || world_mask[world] == 0) return;
  uint mocap_values = nmocap ? batch*nmocap*7 : 1;
  uint tree_offset = mocap_values;
  uint mocap_id_offset = tree_offset+nbody;
  uint awake_offset = mocap_id_offset+nbody;
  uint valid_offset = awake_offset+batch*max(ntree, 1u);
  uint mismatch_offset = valid_offset+batch;
  device const float* geom_pos_low=geom_pos+max(ngeom*3,1u);
  device const float* geom_pos_tail=geom_pos+2*max(ngeom*3,1u);
  device const float* site_pos_low=site_pos+max(nsite*3,1u);
  device const float* site_pos_tail=site_pos+2*max(nsite*3,1u);
  device const float* qpos0_low=qpos0+max(nq,1u);
  device const float* qpos0_tail=qpos0+2*max(nq,1u);
  uint body_q_stride=max(nbody*4,1u);
  uint geom_q_stride=max(ngeom*4,1u);
  uint site_q_stride=max(nsite*4,1u);
  uint inert_q_stride=max(nbody*4,1u);
  uint axis_stride=max(njnt*3,1u);
  device const float* jnt_axis_low=jnt_axis+axis_stride;
  device const float* jnt_axis_tail=jnt_axis+2*axis_stride;
  bool cached = auxiliary[valid_offset+world] != 0.0f;
  uint tree_stride = max(ntree, 1u);
  for (uint t=0; t<tree_stride; ++t) {
    auxiliary[mismatch_offset+world*tree_stride+t] = 0.0f;
  }
  uint qbase = world*nq;
  uint posebase = world*nbody;
  store3(xpos, posebase*3, float3(0));
  store3(xpos_low, posebase*3, float3(0));
  store3(xpos_tail, posebase*3, float3(0));
  store4(xquat, posebase*4, float4(1, 0, 0, 0));
  store4(xquat_low, posebase*4, float4(0));
  store4(xquat_tail, posebase*4, float4(0));
  for (uint b=1; b<nbody; ++b) {
    int tree = int(auxiliary[tree_offset+b]);
    bool sleeping = tree >= 0 && uint(tree) < ntree
        && auxiliary[awake_offset+world*tree_stride+uint(tree)] == 0.0f;
    uint p = uint(parent[b]);
    KinQ parent_q=kinq(load4(xquat,(posebase+p)*4),
                       load4(xquat_low,(posebase+p)*4),
                       load4(xquat_tail,(posebase+p)*4));
    KinDD3 parent_pos=kin_from3(load3(xpos,(posebase+p)*3),
        load3(xpos_low,(posebase+p)*3),load3(xpos_tail,(posebase+p)*3));
    KinDD3 body_local=kin3_load(body_pos,b*3,max(nbody*3,1u));
    KinMat parent_mat=kinq_matrix(parent_q);
    KinDD3 pos_pair=kin3_add(parent_pos,kinmat_mul(parent_mat,body_local));
    float3 pos=kin3_value(pos_pair);
    KinQ quat=kinq_mul(parent_q,kinq_load(body_quat,b*4,body_q_stride));
    int mid = int(auxiliary[mocap_id_offset+b]);
    if (mid >= 0 && uint(mid) < max(nmocap, 1u)) {
      // Pinned MuJoCo 3.10 mj_kinematics: the prescribed mocap pose replaces
      // the compiled body frame, composed with the parent frame (the pinned
      // compiler restricts mocap bodies to jointless world children).
      // Layout: all positions [batch, nmocap, 3] then all quaternions.
      uint nm = max(nmocap, 1u);
      uint mbase = world*nm + uint(mid);
      float3 mpos = load3(auxiliary, mbase*3);
      KinQ mquat=kinq_norm(kinq(load4(auxiliary,batch*nm*3+mbase*4)));
      KinDD3 mocap_local=kin3(mpos,float3(0.0f),float3(0.0f));
      pos_pair=kin3_add(parent_pos,kinmat_mul(parent_mat,mocap_local));
      pos=kin3_value(pos_pair);
      quat=kinq_mul(parent_q,mquat);
    }
    for (uint j=0; j<njnt; ++j) {
      if (uint(jnt_bodyid[j]) != b) continue;
      int typ = jnt_type[j];
      uint qa = qbase + uint(jnt_qposadr[j]);
      KinDD3 joint_local=kin3_load(jnt_pos,j*3,max(njnt*3,1u));
      KinDD3 axis_local=kin3(
          load3(jnt_axis,j*3),load3(jnt_axis_low,j*3),
          load3(jnt_axis_tail,j*3));
      KinDD3 anchor_pair=kin3_add(pos_pair,kinq_rotate(quat,joint_local));
      KinDD3 axis_pair=kinq_rotate(quat,axis_local);
      float3 anchor=kin3_value(anchor_pair);
      float3 axis=kin3_value(axis_pair);
      store3(joint_anchor, (world*njnt+j)*3, anchor);
      store3(joint_axis, (world*njnt+j)*3, axis);
      if (typ == 0) {
        pos = load3(qpos, qa);
        pos_pair=kin3(pos,load3(precision,pb+jnt_qposadr[j]),
                         load3(precision,pb+nq+jnt_qposadr[j]));
        quat = kinq_norm(kinq(load4(qpos, qa+3),load4(precision,pb+jnt_qposadr[j]+3),
             load4(precision,pb+nq+jnt_qposadr[j]+3)));
        store3(joint_anchor, (world*njnt+j)*3, pos);
        store3(joint_axis, (world*njnt+j)*3, load3(jnt_axis, j*3));
        anchor_pair=pos_pair;
        axis_pair=axis_local;
      } else if (typ == 1) {
        quat=kinq_mul(quat,kinq_norm(kinq(load4(qpos,qa),load4(precision,pb+jnt_qposadr[j]),
             load4(precision,pb+nq+jnt_qposadr[j]))));
        pos_pair=kin3_sub(anchor_pair,kinq_rotate(quat,joint_local));
        pos=kin3_value(pos_pair);
      } else if (typ == 2) {
        KinDD delta=kin_sub(KinDD{qpos[qa],precision[pb+jnt_qposadr[j]],precision[pb+nq+jnt_qposadr[j]]},
            KinDD{qpos0[jnt_qposadr[j]],qpos0_low[jnt_qposadr[j]],
                  qpos0_tail[jnt_qposadr[j]]});
        // engine_core_smooth.c transforms the local slide axis to xaxis
        // before mji_addToScl3(xpos, xaxis, delta). Keep that operation order
        // rather than rotating a locally scaled vector.
        pos_pair=KinDD3{kin_fma(axis_pair.x,delta,pos_pair.x),
                        kin_fma(axis_pair.y,delta,pos_pair.y),
                        kin_fma(axis_pair.z,delta,pos_pair.z)};
        pos=kin3_value(pos_pair);
      } else if (typ == 3) {
        KinDD angle=kin_sub(KinDD{qpos[qa],precision[pb+jnt_qposadr[j]],
                                  precision[pb+nq+jnt_qposadr[j]]},
            KinDD{qpos0[jnt_qposadr[j]],qpos0_low[jnt_qposadr[j]],
                  qpos0_tail[jnt_qposadr[j]]});
        KinDD half_angle=kin_scale(angle,0.5f);
        KinDD cosine=kin_sincos(half_angle,false),sine=kin_sincos(half_angle,true);
        KinQ local_rotation=KinQ{cosine,kin_mul(axis_local.x,sine),
            kin_mul(axis_local.y,sine),kin_mul(axis_local.z,sine)};
        quat=kinq_mul(quat,local_rotation);
        pos_pair=kin3_sub(anchor_pair,kinq_rotate(quat,joint_local));
        pos=kin3_value(pos_pair);
      }
      // Keep the ordered joint frame, before subsequent joints on this
      // body rotate or translate it. The public high planes stay unchanged.
      uint joint_offset=(world*njnt+j)*3;
      store3(joint_anchor_low,joint_offset,kin_residual_low(anchor_pair,load3(joint_anchor,joint_offset)));
      store3(joint_anchor_tail,joint_offset,kin_residual_tail(anchor_pair,load3(joint_anchor,joint_offset)));
      store3(joint_axis_low,joint_offset,kin_residual_low(axis_pair,load3(joint_axis,joint_offset)));
      store3(joint_axis_tail,joint_offset,kin_residual_tail(axis_pair,load3(joint_axis,joint_offset)));
    }
    // Pinned mj_kinematics1 computes a candidate pose even for a sleeping
    // articulated body, compares it exactly with the retained pose, and marks
    // the tree awake on any change. The scheduler then wakes its sleep cycle
    // before collision/constraint assembly. Do this comparison before the
    // cached-pose early-out; otherwise state writes made before the first
    // forward pass (and later in-place qpos edits) are silently ignored.
    if (sleeping) {
      bool has_joint = false;
      for (uint j=0; j<njnt; ++j) {
        has_joint = has_joint || uint(jnt_bodyid[j]) == b;
      }
      if (has_joint) {
        KinDD3 old_pos=kin_from3(load3(xpos,(posebase+b)*3),
            load3(xpos_low,(posebase+b)*3),load3(xpos_tail,(posebase+b)*3));
        KinQ old_quat=kinq(load4(xquat,(posebase+b)*4),
            load4(xquat_low,(posebase+b)*4),load4(xquat_tail,(posebase+b)*4));
        bool matches=kin3_equal(pos_pair,old_pos) && kinq_equal(quat,old_quat);
        if (!matches) {
          // The host consumes this float storage through an int32 view, so
          // write the integer-one bit pattern rather than float 1.0.
          auxiliary[mismatch_offset+world*tree_stride+uint(tree)] =
              as_type<float>(1u);
        }
        if (cached && matches) continue;
      }
      // A sleeping body without joints cannot differ from its compiled frame
      // under qpos changes. It still retains its validated frame when cached.
      else if (cached) {
        continue;
      }
    }
    store3(xpos, (posebase+b)*3, pos);
    store3(xpos_low,(posebase+b)*3,kin_residual_low(pos_pair,pos));
    store3(xpos_tail,(posebase+b)*3,kin_residual_tail(pos_pair,pos));
    quat=kinq_norm(quat);
    kinq_store(xquat,xquat_low,xquat_tail,(posebase+b)*4,quat);
  }
  // Match mj_kinematics2's inertial frame SAMEFRAME cases before any geom or
  // site asks to reuse the body's inertial frame.
  for (uint b=0; b<nbody; ++b) {
    int tree=int(auxiliary[tree_offset+b]);
    if (cached && tree>=0
        && auxiliary[awake_offset+world*tree_stride+uint(tree)]==0.0f) continue;
    uint boff=(posebase+b)*4, poff=(posebase+b)*3;
    KinQ bq=kinq(load4(xquat,boff),load4(xquat_low,boff),
                 load4(xquat_tail,boff));
    KinDD3 bp=kin_from3(load3(xpos,poff),load3(xpos_low,poff),
                        load3(xpos_tail,poff));
    KinDD3 local=kin3_load(body_ipos,b*3,max(nbody*3,1u));
    int sf=body_sameframe[b];
    KinQ iq;
    KinDD3 ip;
    if (sf==1) {
      iq=bq; ip=bp;
    } else if (sf==3) {
      iq=bq; ip=kin3_add(bp,kinmat_mul(kinq_matrix(bq),local));
    } else if (sf==2 || sf==4) {
      // The compiler emits these only for frames that alias an already
      // materialized inertial frame. Preserve its cached representation.
      if (!cached) { iq=kinq_mul(bq,kinq_load(body_iquat,b*4,inert_q_stride));
                     ip=kin3_add(bp,kinmat_mul(kinq_matrix(bq),local)); }
      else continue;
    } else {
      iq=kinq_mul(bq,kinq_load(body_iquat,b*4,inert_q_stride));
      ip=kin3_add(bp,kinmat_mul(kinq_matrix(bq),local));
    }
    float3 ih=kin3_value(ip);
    store3(inertial_xpos,poff,ih);
    store3(inertial_xpos_low,poff,kin_residual_low(ip,ih));
    store3(inertial_xpos_tail,poff,kin_residual_tail(ip,ih));
    kinq_store(inertial_xquat,inertial_xquat_low,inertial_xquat_tail,boff,iq);
  }
  for (uint g=0; g<ngeom; ++g) {
    uint b = uint(geom_bodyid[g]);
    int tree = int(auxiliary[tree_offset+b]);
    if (cached && tree >= 0 && auxiliary[awake_offset+world*tree_stride+uint(tree)] == 0.0f) continue;
    KinQ bq=kinq(load4(xquat,(posebase+b)*4),
                 load4(xquat_low,(posebase+b)*4),
                 load4(xquat_tail,(posebase+b)*4));
    uint geometry = (world*ngeom+g)*3;
    KinDD3 body_position = kin_from3(
        load3(xpos,(posebase+b)*3), load3(xpos_low,(posebase+b)*3),
        load3(xpos_tail,(posebase+b)*3));
    KinDD3 local_position = kin_from3(
        load3(geom_pos,g*3), load3(geom_pos_low,g*3),
        load3(geom_pos_tail,g*3));
    uint io=(posebase+b)*3;
    KinDD3 inertial_position=kin_from3(load3(inertial_xpos,io),
        load3(inertial_xpos+max(batch*nbody*3,1u),io),
        load3(inertial_xpos+2*max(batch*nbody*3,1u),io));
    KinQ iq=kinq(load4(inertial_xquat,(posebase+b)*4),
        load4(inertial_xquat_low,(posebase+b)*4),
        load4(inertial_xquat_tail,(posebase+b)*4));
    int sf=geom_sameframe[g];
    KinDD3 origin=body_position;
    KinMat frame_mat=kinq_matrix(bq);
    KinQ gq;
    KinDD3 exact_geom_position;
    if (sf==1) { exact_geom_position=body_position; gq=bq; }
    else if (sf==2) { exact_geom_position=inertial_position; gq=iq; }
    else if (sf==4) {
      // mj_local2Global uses body xmat for INERTIAROT position and ximat
      // only for orientation. This differs from INERTIA (code 2).
      origin=body_position; frame_mat=kinq_matrix(bq);
      exact_geom_position=kin3_add(origin,kinmat_mul(frame_mat,local_position));
      gq=iq;
    } else if (sf==3) {
      exact_geom_position=kin3_add(origin,kinmat_mul(frame_mat,local_position));
      gq=bq;
    } else {
      exact_geom_position=kin3_add(origin,kinmat_mul(frame_mat,local_position));
      gq=kinq_mul(bq,kinq_load(geom_quat,g*4,geom_q_stride));
    }
    float3 rounded_geom_position=kin3_value(exact_geom_position);
    store3(geom_xpos, (world*ngeom+g)*3,
           rounded_geom_position);
    store3(geom_xpos_low,geometry,
           kin_residual_low(exact_geom_position,rounded_geom_position));
    store3(geom_xpos_tail,geometry,
           kin_residual_tail(exact_geom_position,rounded_geom_position));
    uint gqoff=(world*ngeom+g)*4;
    kinq_store(geom_xquat,geom_xquat_low,geom_xquat_tail,gqoff,gq);
    KinMat gm=(sf==3) ? kinq_matrix(bq) : (sf==4 ? kinq_matrix(iq) : kinq_matrix(gq));
    kinmat_store(geom_xmat,geom_xmat_low,geom_xmat_tail,
                 (world*ngeom+g)*9,gm);
  }
  for (uint s=0; s<nsite; ++s) {
    uint b = uint(site_bodyid[s]);
    int tree = int(auxiliary[tree_offset+b]);
    if (cached && tree >= 0 && auxiliary[awake_offset+world*tree_stride+uint(tree)] == 0.0f) continue;
    KinQ bq=kinq(load4(xquat,(posebase+b)*4),
                 load4(xquat_low,(posebase+b)*4),
                 load4(xquat_tail,(posebase+b)*4));
    KinDD3 body_position=kin_from3(load3(xpos,(posebase+b)*3),
        load3(xpos_low,(posebase+b)*3),load3(xpos_tail,(posebase+b)*3));
    uint io=(posebase+b)*3;
    KinDD3 inertial_position=kin_from3(load3(inertial_xpos,io),
        load3(inertial_xpos+max(batch*nbody*3,1u),io),
        load3(inertial_xpos+2*max(batch*nbody*3,1u),io));
    KinQ iq=kinq(load4(inertial_xquat,(posebase+b)*4),
        load4(inertial_xquat_low,(posebase+b)*4),
        load4(inertial_xquat_tail,(posebase+b)*4));
    KinDD3 local_position=KinDD3{
        KinDD{load3(site_pos,s*3).x,load3(site_pos_low,s*3).x,load3(site_pos_tail,s*3).x},
        KinDD{load3(site_pos,s*3).y,load3(site_pos_low,s*3).y,load3(site_pos_tail,s*3).y},
        KinDD{load3(site_pos,s*3).z,load3(site_pos_low,s*3).z,load3(site_pos_tail,s*3).z}};
    int sf=site_sameframe[s];
    KinDD3 site_position; KinQ sq;
    if (sf==1) { site_position=body_position; sq=bq; }
    else if (sf==2) { site_position=inertial_position; sq=iq; }
    else if (sf==4) {
      site_position=kin3_add(body_position,
          kinmat_mul(kinq_matrix(bq),local_position)); sq=iq;
    } else if (sf==3) {
      site_position=kin3_add(body_position,
          kinmat_mul(kinq_matrix(bq),local_position)); sq=bq;
    } else {
      site_position=kin3_add(body_position,
          kinmat_mul(kinq_matrix(bq),local_position));
      sq=kinq_mul(bq,kinq_load(site_quat,s*4,site_q_stride));
    }
    uint soff=(world*nsite+s)*3;
    float3 site_high=kin3_value(site_position);
    store3(site_xpos,soff,site_high);
    store3(site_xpos_low,soff,kin_residual_low(site_position,site_high));
    store3(site_xpos_tail,soff,kin_residual_tail(site_position,site_high));
    kinq_store(site_xquat,site_xquat_low,site_xquat_tail,
               (world*nsite+s)*4,sq);
  }
  auxiliary[valid_offset+world] = 1.0f;
}
