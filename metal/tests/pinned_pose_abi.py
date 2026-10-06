"""Construct the complete retained pose ABI from independent CPU oracle data.

This is a test fixture, never a production physics fallback. Keep all three
binary32 words of each binary64 field rather than inventing zero residuals.
"""
import mujoco
import numpy as np


def pinned_pose_abi(data, device="cpu"):
  import torch

  def tensor(value):
    return torch.tensor(np.asarray(value).copy()[None], dtype=torch.float32,
                        device=device)

  def words(value):
    value = np.asarray(value, dtype=np.float64)
    high = value.astype(np.float32)
    remainder = value - high.astype(np.float64)
    low = remainder.astype(np.float32)
    tail = (remainder - low.astype(np.float64)).astype(np.float32)
    return tensor(high), tensor(low), tensor(tail)

  result = {"joint_anchor": tensor(data.xanchor),
            "joint_axis": tensor(data.xaxis)}
  values = {"body_pos": data.xpos, "body_quat": data.xquat,
            "inertial_pos": data.xipos, "geom_pos": data.geom_xpos,
            "site_pos": data.site_xpos, "geom_xmat": data.geom_xmat}
  for prefix, matrices in (("inertial", data.ximat),
                           ("geom", data.geom_xmat),
                           ("site", data.site_xmat)):
    quat = np.empty((len(matrices), 4))
    for index, matrix in enumerate(matrices):
      mujoco.mju_mat2Quat(quat[index], matrix)
    values[prefix + "_quat"] = quat
  for name, value in values.items():
    result[name], result[name + "_low"], result[name + "_tail"] = words(value)
  return result
