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

"""CPU preflight, model inspection, and opt-in Metal diagnostics."""

import argparse
from dataclasses import asdict
from enum import Enum
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys

import numpy as np

import mujoco

from mujoco_metal import __version__
from mujoco_metal.model import load_model
from mujoco_metal.registry import FEATURES
from mujoco_metal.registry import INVENTORY_COMPLETE
from mujoco_metal.registry import REQUIREMENTS
from mujoco_metal.registry import TARGET_MUJOCO_VERSION

_INSTALL_COMMAND = "python -m pip install -e ./metal[metal,test]"


def _jsonable(value):
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, dict):
    return {key: _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  return value


def preflight(model_path=None, include_inventory=False):
  package = Path(__file__).resolve().parent
  shader = package / "shaders" / "kinematics.metal"
  # Inventory the installed package, including development kernels. A legacy
  # hardcoded list hid component solves, deformables and query operators from
  # source/wheel provenance reports. This is read-only, never a GPU test.
  shader_paths = {path.stem: path for path in sorted(
      (package / "shaders").glob("*.metal"))}
  requirement_counts = {}
  for requirement in REQUIREMENTS:
    key = f"{requirement.implementation.value}/{requirement.qualification.value}"
    requirement_counts[key] = requirement_counts.get(key, 0) + 1
  result = {
      "package_version": __version__,
      "package_path": str(package),
      "target_mujoco_version": TARGET_MUJOCO_VERSION,
      "actual_mujoco_version": mujoco.__version__,
      "shader_path": str(shader),
      "shader_sha256": hashlib.sha256(shader.read_bytes()).hexdigest(),
      "actuation_shader": {
          "path": str(package / "shaders" / "actuation.metal"),
          "sha256": hashlib.sha256(
              (package / "shaders" / "actuation.metal").read_bytes()
          ).hexdigest(),
      },
      "shaders": {
          name: {
              "path": str(path),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          }
          for name, path in shader_paths.items()
      },
      "inventory_complete": INVENTORY_COMPLETE,
      "diagnostic_execution": "host_inventory_only",
      "requirement_counts": requirement_counts,
      "requirement_counts_scope": (
          "inventory rows, including host utilities and enum sentinels; "
          "not a physics completion percentage or a test of this installation"
      ),
      "gpu_qualified": False,
      "gpu_qualification_scope": (
          "complete physics backend; per-stage narrow results are listed in "
          "feature inventory"
      ),
      "stages": {
          "model_inspection": "CPU",
          "kinematics": (
              "CPU oracle; Metal narrowly GPU-qualified on M1 fixtures"
          ),
          "dynamics": (
              "CPU smooth M/bias oracle; Metal narrowly GPU-qualified on M1 "
              "fixtures"
          ),
          "acceleration_solve": (
              "native dense SPD solve; narrowly GPU-qualified on M1 "
              "synthetic systems"
          ),
          "device_state": (
              "persistent MPS state with host reset/checkpoint lifecycle; "
              "pipeline qualification separate"
          ),
          "integration": (
              "native semi-implicit Euler; narrowly GPU-qualified against "
              "MuJoCo 3.10 on M1 fixtures"
          ),
          "contact_free_euler_v1": (
              "narrowly GPU-qualified on M1 fixtures; bounded profile only"
          ),
          "contact_free_forces_euler_v1": (
              "narrowly GPU-qualified on M1 with 36 trajectories and 10 "
              "GPU regression cases; applied generalized forces and linear "
              "joint damping"
          ),
          "scalar_motor_force": (
              "narrowly GPU-qualified on M1 fixtures; scalar joint force "
              "stage only"
          ),
          "contact_free_motor_euler_v1": (
              "narrowly GPU-qualified on M1; scalar hinge/slide motors, "
              "clipping and disable flags; 15 independent 1000-step trajectories"
          ),
          "full_stepping": "explicit bounded and development integrated profiles; broad compositions unqualified; see DEVELOPMENT.md",
          "passive": "narrowly qualified rigid springs, polynomial damping, Cartesian wrenches and gravcomp",
          "transmissions": "narrowly qualified stateless scalar servos and fixed-joint tendons",
          "collision": "qualified primitive subsets; broader rigid/flex development paths have open correction gates",
          "constraints": "coupled scalar/equality/contact paths; sparse constraint Jacobian missing and broad solver qualification open",
          "fluid": "inertia-box and geom-fluid development forces; composition qualification separate",
          "implicitfast": "eligible free-body midpoint and development integrated implicit/implicitfast paths; broad compositions unqualified",
          "sensors": "current-state queries and development stored step-stage sensors; family/composition limits in coverage inventory",
          "rendering": "upstream host OpenGL; no native Metal renderer",
      },
  }
  if model_path:
    model = load_model(Path(model_path))
    result["model"] = {
        name: getattr(model, name)
        for name in (
            "nq",
            "nv",
            "nu",
            "nbody",
            "njnt",
            "ngeom",
            "nsite",
            "ntendon",
            "nmocap",
        )
    }
  if include_inventory:
    from mujoco_metal.binding_inventory import pinned_enum_inventory, pinned_binding_surface
    result["pinned_enums"] = pinned_enum_inventory()
    result["pinned_binding_surface"] = pinned_binding_surface()
    result["features"] = [_jsonable(asdict(row)) for row in FEATURES]
    result["requirements"] = [_jsonable(asdict(row)) for row in REQUIREMENTS]
  return result


def _installed_version(distribution):
  try:
    return importlib.metadata.version(distribution)
  except importlib.metadata.PackageNotFoundError:
    return None


def doctor(gpu=False):
  """Collect diagnostics without importing PyTorch unless --gpu is requested."""
  python_version = platform.python_version()
  torch_version = _installed_version("torch")
  result = {
      "command": "doctor",
      "package_version": __version__,
      "platform": {
          "system": platform.system(),
          "release": platform.release(),
          "machine": platform.machine(),
          "python_implementation": platform.python_implementation(),
      },
      "python": {
          "version": python_version,
          "executable": sys.executable,
          "supported": sys.version_info[:2] == (3, 12),
          "required": "3.12.x",
      },
      "mujoco": {
          "version": mujoco.__version__,
          "required": TARGET_MUJOCO_VERSION,
          "compatible": mujoco.__version__ == TARGET_MUJOCO_VERSION,
      },
      "torch": {
          "installed_version": torch_version,
          "required_version": "2.9.1",
          "optional_extra": "metal",
          "imported": False,
      },
      "environment": {
          "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get(
              "PYTORCH_ENABLE_MPS_FALLBACK", "<unset>"
          ),
          "MUJOCO_METAL_RUN_GPU": os.environ.get(
              "MUJOCO_METAL_RUN_GPU", "<unset>"
          ),
          "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", "<unset>"),
          "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", "<unset>"),
      },
      "gpu": {
          "requested": bool(gpu),
          "tested": False,
          "passed": None,
          "status": "not_requested" if not gpu else "not_run",
      },
  }
  result["ok"] = bool(
      result["python"]["supported"] and result["mujoco"]["compatible"]
  )
  if gpu:
    _run_gpu_doctor(result)
    result["ok"] = bool(result["ok"] and result["gpu"]["passed"])
  return result


def _run_gpu_doctor(result):
  gpu = result["gpu"]
  gpu["passed"] = False
  if not result["python"]["supported"]:
    gpu.update(
        status="unsupported_python",
        error="The native smoke check is qualified for Python 3.12.x.",
        action=(
            "Create a Python 3.12 environment, reinstall the pinned metal "
            "extra, and retry."
        ),
    )
    return
  if not result["mujoco"]["compatible"]:
    gpu.update(
        status="mujoco_version_mismatch",
        error=(
            f"The native smoke check requires MuJoCo {TARGET_MUJOCO_VERSION}; "
            f"found {mujoco.__version__}."
        ),
        action=(
            "Reinstall the package in a clean environment so pip installs "
            "its pinned MuJoCo version."
        ),
    )
    return
  if result["platform"]["system"] != "Darwin":
    gpu.update(
        status="unsupported_platform",
        error="Native Metal requires macOS on Apple Silicon.",
        action="Run this command on an Apple Silicon Mac with macOS.",
    )
    return
  if result["platform"]["machine"].lower() not in ("arm64", "aarch64"):
    gpu.update(
        status="unsupported_architecture",
        error="Native Metal requires an Apple Silicon (arm64) Mac.",
        action="Use Python running natively on Apple Silicon.",
    )
    return
  fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
  if fallback == "1":
    gpu.update(
        status="mps_fallback_enabled",
        error="PYTORCH_ENABLE_MPS_FALLBACK=1 can hide unsupported MPS operations.",
        action="Unset PYTORCH_ENABLE_MPS_FALLBACK or set it to 0, then retry.",
    )
    return
  if result["torch"]["installed_version"] is None:
    gpu.update(
        status="torch_missing",
        error="PyTorch is not installed.",
        action=f"Install the metal extra with: {_INSTALL_COMMAND}",
    )
    return
  if result["torch"]["installed_version"] != "2.9.1":
    gpu.update(
        status="torch_version_mismatch",
        error=(
            "The project currently qualifies PyTorch 2.9.1; found "
            f"{result['torch']['installed_version']}."
        ),
        action=(
            "Install the pinned metal extra in a clean Python 3.12 "
            "environment, then retry."
        ),
    )
    return
  try:
    import torch

    result["torch"]["imported"] = True
    gpu["torch_mps_built"] = bool(torch.backends.mps.is_built())
    gpu["torch_mps_available"] = bool(torch.backends.mps.is_available())
    if not gpu["torch_mps_built"] or not gpu["torch_mps_available"]:
      gpu.update(
          status="mps_unavailable",
          error="PyTorch cannot access the Metal Performance Shaders device.",
          action=(
              "Check native arm64 macOS, update macOS, and reinstall the "
              "pinned torch==2.9.1 metal extra."
          ),
      )
      return

    import mujoco_metal

    xml = """<mujoco><option timestep=".002" gravity="0 0 0">
      <flag contact="disable"/></option><worldbody>
      <body><joint type="hinge" axis="0 1 0" damping=".1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".05" mass="1"/>
      </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    simulation = mujoco_metal.MetalSimulation(
        model, profile="contact_free_forces_euler_v1"
    )
    applied = np.array([[0.2]], dtype=np.float32)
    gpu["tested"] = True
    status = simulation.step(steps=2, qfrc_applied=applied)
    torch.mps.synchronize()
    status_values = status.detach().cpu().numpy()
    snapshot = simulation.state.snapshot()
    finite = bool(
        np.all(np.isfinite(snapshot.qpos))
        and np.all(np.isfinite(snapshot.qvel))
        and np.all(np.isfinite(snapshot.qacc))
        and np.all(np.isfinite(snapshot.time))
    )
    if not finite or np.any(status_values != 0) or np.any(snapshot.status != 0):
      gpu.update(
          status="smoke_check_failed",
          passed=False,
          error=(
              "The native stepping smoke check returned a nonzero status or "
              "nonfinite state."
          ),
          action=(
              "Check the Metal toolchain and macOS runtime; see metal/INSTALL.md "
              "and retry with PYTORCH_ENABLE_MPS_FALLBACK=0."
          ),
          step_status=status_values.tolist(),
          state_status=snapshot.status.tolist(),
          finite_state=finite,
      )
      return
    gpu.update(
        status="passed",
        passed=True,
        profile="contact_free_forces_euler_v1",
        steps=2,
        synchronized=True,
        finite_state=True,
        status_codes=status_values.tolist(),
    )
  except Exception as error:  # Surface device failures without a traceback.
    gpu.update(
        status=(
            "smoke_check_failed" if gpu["tested"] else "initialization_failed"
        ),
        passed=False,
        error=f"{type(error).__name__}: {error}",
        action=(
            "Check Xcode Command Line Tools and the macOS Metal runtime, "
            "confirm the pinned environment, and retry. See metal/INSTALL.md."
        ),
    )


def _print_doctor(result, as_json):
  if as_json:
    print(json.dumps(result, indent=2, sort_keys=True))
    return
  print(f"MuJoCo Metal doctor: {'OK' if result['ok'] else 'CHECK REQUIRED'}")
  print(f"  package: {result['package_version']}")
  print(
      f"  platform: {result['platform']['system']} "
      f"{result['platform']['release']} "
      f"({result['platform']['machine']})"
  )
  print(
      f"  Python: {result['python']['version']} "
      f"({'supported' if result['python']['supported'] else 'requires 3.12.x'})"
  )
  print(
      f"  MuJoCo: {result['mujoco']['version']} "
      f"({'supported' if result['mujoco']['compatible'] else 'version mismatch'})"
  )
  if not result["python"]["supported"]:
    print("  action: Create an environment with Python 3.12.x and rerun.")
  if not result["mujoco"]["compatible"]:
    print(
        "  action: Reinstall the project-pinned MuJoCo version in this "
        "environment."
    )
  torch = result["torch"]
  print(
      f"  PyTorch: {torch['installed_version'] or 'not installed'} "
      f"(optional metal extra; import {'performed' if torch['imported'] else 'skipped'})"
  )
  print(
      "  MPS fallback: "
      f"{result['environment']['PYTORCH_ENABLE_MPS_FALLBACK']}"
  )
  gpu = result["gpu"]
  print(f"  GPU smoke test: {gpu['status']}")
  if gpu.get("error"):
    print(f"  detail: {gpu['error']}")
    print(f"  action: {gpu['action']}")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  preflight_parser = subparsers.add_parser("preflight")
  preflight_parser.add_argument("--model", type=Path)
  preflight_parser.add_argument("--json", action="store_true", dest="as_json")
  preflight_parser.add_argument("--inventory", action="store_true")
  doctor_parser = subparsers.add_parser(
      "doctor", help="Check the install and optionally smoke-test native Metal"
  )
  doctor_parser.add_argument(
      "--gpu", action="store_true", help="Run a native MPS stepping smoke test"
  )
  doctor_parser.add_argument("--json", action="store_true", dest="as_json")
  coverage_parser = subparsers.add_parser(
      "coverage", help="Print the machine-readable support-contract table"
  )
  coverage_parser.add_argument("--json", action="store_true", dest="as_json")
  demo_parser = subparsers.add_parser(
      "demo", help="Run bundled demo scripts and their model assets")
  demo_parser.add_argument("--list", action="store_true", dest="list_demos")
  demo_parser.add_argument("name", nargs="?")
  demo_parser.add_argument("demo_args", nargs=argparse.REMAINDER)
  args = parser.parse_args(argv)
  if args.command == "demo":
    from mujoco_metal.demo_cli import available_demos, run_demo
    if args.list_demos:
      if args.name or args.demo_args:
        demo_parser.error("--list does not accept a demo name or arguments")
      print("\n".join(available_demos()))
      return 0
    if args.name is None:
      demo_parser.error("provide a demo name, or use --list")
    try:
      run_demo(args.name, args.demo_args)
    except ValueError as error:
      demo_parser.error(str(error))
    return 0
  if args.command == "coverage":
    from mujoco_metal.registry import REQUIREMENTS, coverage_table
    from mujoco_metal.registry import _jsonable_requirement
    if args.as_json:
      print(json.dumps([_jsonable_requirement(r) for r in REQUIREMENTS], indent=2, sort_keys=True))
    else:
      print(coverage_table(), end="")
    return 0
  if args.command == "doctor":
    result = doctor(args.gpu)
    _print_doctor(result, args.as_json)
    return 0 if result["ok"] else 1
  result = preflight(args.model, args.inventory)
  if args.as_json:
    print(json.dumps(result, indent=2, sort_keys=True))
  else:
    for key, value in result.items():
      if key != "features":
        print(f"{key}: {value}")
    if args.inventory:
      print(f"features: {len(result['features'])} inventory rows")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
