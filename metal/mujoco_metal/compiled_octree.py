# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Immutable compiled octree operands shared by native SDF query consumers."""

import numbers

import numpy as np


def compiled_octree_block(model, adr, count):
  """Pack children and all three source-word planes for the 50-word decoder.

  This is model preprocessing only. Distances and gradients execute on device.
  Array planes are children[8N], AABB[6N], coefficients[8N], followed by
  AABB/coefficient low words and then their tail words.
  """
  from mujoco_metal.model import require_triple_word_domain
  if (isinstance(adr, bool) or not isinstance(adr, numbers.Integral)
      or isinstance(count, bool) or not isinstance(count, numbers.Integral)
      or adr < 0 or not 0 < count < 2**24
      or adr + count > len(model.oct_child)):
    raise ValueError("invalid compiled octree node range")
  child = np.asarray(model.oct_child[adr:adr+count]).reshape(-1)
  aabb = require_triple_word_domain(model.oct_aabb[adr:adr+count],
                                   what="compiled octree AABB")
  coeff = require_triple_word_domain(model.oct_coeff[adr:adr+count],
                                    what="compiled octree coefficients")
  if child.size != 8*count or aabb.size != 6*count or coeff.size != 8*count:
    raise ValueError("invalid compiled octree layout")
  if np.any(child < -1) or np.any(child >= 2**24):
    raise ValueError("compiled octree child indices must be float32-exact")
  high = [v.astype(np.float32) for v in (aabb, coeff)]
  low = [(v-h.astype(np.float64)).astype(np.float32)
         for v, h in zip((aabb, coeff), high)]
  tail = [(v-h.astype(np.float64)-l.astype(np.float64)).astype(np.float32)
          for v, h, l in zip((aabb, coeff), high, low)]
  return np.concatenate([child.astype(np.float32), *high, *low, *tail])
