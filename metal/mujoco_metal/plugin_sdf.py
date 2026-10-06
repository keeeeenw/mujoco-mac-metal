# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device query for the five bundled MuJoCo 3.10.0 analytic SDF plugins.

The inline shader functions are also the candidate-loop ABI: callers can
concatenate ``shaders/plugin_sdf.metal`` into an existing fixed-shape contact
shader and invoke ``plugin_sdf_distance`` plus
``plugin_sdf_gradient(kind, point, attributes_high, attributes_low)`` directly.
The low array carries the residual of source decimal attributes after their
float32 high part, preserving branch-sensitive values without host callbacks.
"""

from pathlib import Path
import math
import numbers

import numpy as np

from mujoco_metal.bundled_plugins import BundledPluginModel


_SHADER = Path(__file__).parent / "shaders" / "plugin_sdf.metal"
_KIND_NAMES = {
    1: "bolt", 2: "bowl", 3: "gear", 4: "nut", 5: "torus",
}
_PI = math.pi


def _check_query(kind, point, attributes):
  if isinstance(kind, (bool, np.bool_)) or not isinstance(kind, numbers.Integral):
    raise TypeError("kind must be an integer SDF kind")
  kind = int(kind)
  if kind not in _KIND_NAMES:
    raise ValueError("kind must name one of the five bundled SDF plugins")
  p = np.asarray(point, dtype=np.float64)
  a = np.asarray(attributes, dtype=np.float64)
  if p.shape != (3,) or a.shape != (5,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(a)):
    raise ValueError("point and attributes must be finite [3] and [5] vectors")
  return kind, p, a


def sdf_distance_local(kind, point, attributes) -> float:
  """Source-shaped CPU implementation used for construction/oracle tests.

  ``kind`` uses the stable IDs in ``BundledPluginModel.instance_kind``.
  This routine is not called by native simulation or contact dispatch.
  """
  kind, p, a = _check_query(kind, point, attributes)
  x, y, z = (float(v) for v in p)
  root2 = math.sqrt(2.0) * .5
  if kind == 5:
    return math.sqrt((math.hypot(x, y) - a[0]) ** 2 + z * z) - a[1]
  if kind == 2:
    height, radius, thick = a[:3]
    width = math.sqrt(max(radius * radius - height * height, 0.0))
    q0, q1 = math.hypot(x, y), z
    if height * q0 < width * q1:
      return math.hypot(q0 - width, q1 - height) - thick
    return abs(math.hypot(q0, q1) - radius) - thick
  if kind in (1, 4):
    radius = a[0]
    rad = math.hypot(x, y) - radius
    azimuth = math.atan2(y, x)
    triangle = abs((z * 12.0 - azimuth / _PI / 2.0)
                   - math.floor(z * 12.0 - azimuth / _PI / 2.0) - .5)
    if kind == 1:
      thread = (rad - triangle / 12.0) * root2
      bolt = max(thread, -(0.5 - abs(z + .5)))
      cone = (z - rad) * root2
      bolt = max(bolt, -(cone + root2))
      k = 6.0 / _PI / 2.0
      angle = -math.floor(azimuth * k + .5) / k
      s0, s1 = math.sin(angle), math.sin(angle + _PI * .5)
      xr = s1 * x - s0 * y
      head = xr - .5
      head = max(head, abs(z + .25) - .25)
      head = max(head, (z + rad - .22) * root2)
      return min(bolt, head)
    thread = (rad - triangle / 12.0) * root2
    cone = (z - rad) * root2
    hole = max(thread, -(cone + .5 * root2))
    hole = min(hole, -cone - .05 * root2)
    k = 6.0 / _PI / 2.0
    angle = -math.floor(azimuth * k + .5) / k
    s0, s1 = math.sin(angle), math.sin(angle + _PI * .5)
    xr = s1 * x - s0 * y
    head = xr - .5
    head = max(head, abs(z + .25) - .25)
    head = max(head, (z + rad - .22) * root2)
    return max(head, -hole)
  # Gear source port.
  alpha, diameter, teeth, thickness, innerdiameter = a
  psi = 3.096e-5 * teeth * teeth - 6.557e-3 * teeth + .551
  R = diameter * .5
  rho = math.hypot(x, y)
  Pd = teeth / diameter
  pitch = _PI / Pd
  addendum = 1.0 / Pd
  Ro = (diameter + 2.0 * addendum) * .5
  h = 2.2 / Pd
  innerR = Ro - h - .14 * diameter
  if innerdiameter >= 0:
    innerR = innerdiameter * .5
  if innerR - rho > 0:
    d2 = innerR - rho
  elif Ro - rho < -.2:
    d2 = rho - Ro
  else:
    Rb = diameter * math.cos(psi) * .5
    fi = math.atan2(y, x) + alpha
    stride = pitch / R
    inv_alpha = math.acos(max(-1.0, min(1.0, Rb / R)))
    inv_phi = math.tan(inv_alpha) - inv_alpha
    shift = stride * .5 - 2.0 * inv_phi
    fia = ((fi + shift * .5) - stride * math.floor((fi + shift * .5) / stride)) - shift * .5
    fib0 = -fi - shift + shift * .5
    fib = fib0 - stride * math.floor(fib0 / stride) - shift * .5
    da = db = -1e6
    if Rb < rho:
      arc = math.acos(max(-1.0, min(1.0, Rb / rho)))
      ta = math.sqrt(max(rho * rho - Rb * Rb, 0.0))
      da, db = ta - Rb * (fia + arc), ta - Rb * (fib + arc)
    gear_outer = rho - Ro
    gear_low = rho - Ro + h
    crown = rho - innerR
    cogs = max(da, db)
    walls = max(fia - (stride - shift), fib - (stride - shift))
    cogs = max(walls, cogs)
    # smoothIntersection(a,b,k)=Subtraction(Intersection(a,b),
    # smoothUnion(Subtraction(a,b),Subtraction(b,a),k)).
    def smooth_union(a0, b0, k0):
      ratio = (b0 - a0) / k0
      hh = min(1.0, max(0.0, .5 + .5 * ratio))
      return b0 * (1.0 - hh) + a0 * hh - k0 * hh * (1.0 - hh)
    def smooth_intersection(a0, b0, k0):
      su = smooth_union(max(a0, -b0), max(b0, -a0), k0)
      return max(max(a0, b0), -su)
    cogs = smooth_intersection(gear_outer, cogs, .0035 * diameter)
    cogs = smooth_union(gear_low, cogs, Rb - Ro + h)
    d2 = max(cogs, -crown)
  return min(max(d2, abs(z) - thickness * .5), 0.0) + math.hypot(max(d2, 0.0), max(abs(z) - thickness * .5, 0.0))


def sdf_gradient_local(kind, point, attributes) -> np.ndarray:
  """Compute the pinned one-sided gradient for CPU construction/oracle tests.

  The registered callbacks perturb one coordinate at a time by ``1e-8`` in
  ``mjtNum`` (float64).  Native callers use the forward-mode MSL derivative
  instead; this host helper is intentionally not used in simulation.
  """
  kind, p, a = _check_query(kind, point, attributes)
  if kind == 5:
    lenxy = math.hypot(float(p[0]), float(p[1]))
    q = lenxy - float(a[0])
    lenqz = math.hypot(q, float(p[2]))
    den = max(lenqz, np.finfo(np.float64).tiny)
    with np.errstate(invalid="ignore", divide="ignore"):
      return np.asarray([q * p[0] / lenxy / den,
                         q * p[1] / lenxy / den, p[2] / den], dtype=np.float32)
  # Match the upstream callback literally.  Do not use a central difference:
  # the one-sided step defines branch selection at the min/max/floor creases.
  grad = np.empty((3,), dtype=np.float64)
  for axis in range(3):
    delta = np.zeros((3,), dtype=np.float64)
    eps = 1.0e-8
    delta[axis] = eps
    base = sdf_distance_local(kind, p, a)
    grad[axis] = (sdf_distance_local(kind, p + delta, a) - base) / float(eps)
  return grad.astype(np.float32)


class MetalPluginSDFQuery:
  """Fixed-shape native local-coordinate queries for bundled plugin SDFs."""

  def __init__(self, descriptor: BundledPluginModel, *, batch_size: int,
               candidate_capacity: int, device=None):
    if not isinstance(descriptor, BundledPluginModel):
      raise TypeError("descriptor must be a BundledPluginModel")
    for name, value in (("batch_size", batch_size),
                        ("candidate_capacity", candidate_capacity)):
      if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
      if int(value) <= 0 or int(value) > (1 << 31) - 1:
        raise ValueError(f"{name} must be in [1, int32_max]")
    self.batch_size = int(batch_size)
    self.candidate_capacity = int(candidate_capacity)
    if self.batch_size * self.candidate_capacity > (1 << 31) - 1:
      raise ValueError("SDF candidate dispatch exceeds signed int32 capacity")
    if self.batch_size * self.candidate_capacity * 3 > (1 << 31) - 1:
      raise ValueError("SDF gradient dispatch exceeds signed int32 capacity")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("PyTorch MPS compile_shader is required for plugin SDF queries")
    self._torch = torch
    self._device = torch.device("mps") if device is None else torch.device(device)
    if self._device.type != "mps":
      raise ValueError("MetalPluginSDFQuery requires an MPS device")
    self._descriptor = descriptor
    nplugin = len(descriptor.instances)
    if nplugin * 5 > (1 << 31) - 1:
      raise ValueError("SDF attribute table exceeds signed int32 indexing")
    if nplugin == 0:
      # MSL cannot bind a zero-length device allocation; dims still carries
      # logical nplugin=0 so every nonnegative candidate fails closed.
      kinds = np.zeros((1,), dtype=np.int32)
      attrs = np.zeros((1, 5), dtype=np.float32)
      attrs_low = np.zeros((1, 5), dtype=np.float32)
    else:
      kinds = descriptor.instance_kind
      attrs = descriptor.plugin_attributes
      attrs_low = descriptor.plugin_attributes_low
    if len(kinds) > (1 << 31) - 1:
      raise ValueError("plugin instance table exceeds signed int32 indexing")
    self._nplugin = nplugin
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.query_bundled_plugin_sdf
    self._instance_kind = torch.as_tensor(np.array(kinds, copy=True),
                                          dtype=torch.int32, device=self._device)
    self._attributes = torch.as_tensor(np.array(attrs, copy=True),
                                       dtype=torch.float32, device=self._device)
    self._attributes_low = torch.as_tensor(np.array(attrs_low, copy=True),
                                           dtype=torch.float32, device=self._device)
    self._dims = torch.tensor([self.batch_size, self.candidate_capacity, nplugin],
                              dtype=torch.int32, device=self._device)
    self._distance = torch.empty((self.batch_size, self.candidate_capacity),
                                 dtype=torch.float32, device=self._device)
    self._gradient = torch.empty((self.batch_size, self.candidate_capacity, 3),
                                 dtype=torch.float32, device=self._device)
    self._status = torch.empty((self.batch_size, self.candidate_capacity),
                               dtype=torch.int32, device=self._device)

  @property
  def shader_source(self) -> str:
    """Inline source for integration into an existing candidate kernel."""
    return _SHADER.read_text()

  def run_device(self, local_points, plugin_instance):
    """Evaluate padded candidate points and return borrowed MPS output views.

    ``plugin_instance`` is the compiled ``model.geom_plugin[geom_id]`` value
    gathered for each candidate. Negative IDs mean a non-plugin candidate and
    produce zero distance/gradient with status 0. A nonnegative out-of-range ID
    produces status 1. The caller folds status into its candidate/world status.
    """
    torch = self._torch
    expected_points = (self.batch_size, self.candidate_capacity, 3)
    expected_instances = (self.batch_size, self.candidate_capacity)
    for value, shape, dtype, name in (
        (local_points, expected_points, torch.float32, "local_points"),
        (plugin_instance, expected_instances, torch.int32, "plugin_instance"),
    ):
      if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
      if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
      if value.device.type != "mps" or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous on MPS")
    self._kernel(local_points, plugin_instance, self._instance_kind,
                 self._attributes, self._distance, self._gradient, self._status,
                 self._dims, self._attributes_low,
                 threads=(self.batch_size * self.candidate_capacity,),
                 group_size=(1,))
    return {"distance": self._distance, "gradient": self._gradient,
            "status": self._status}
