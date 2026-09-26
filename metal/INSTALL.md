# Install mujoco-mac-metal

The experimental [0.4.0 release](https://pypi.org/project/mujoco-mac-metal/0.4.0/)
is available on PyPI. Use a fresh Python 3.12 environment on an Apple Silicon Mac:

```sh
python3.12 -m venv .venv-metal
source .venv-metal/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade mujoco-mac-metal
```

On macOS arm64, the default dependencies include PyTorch 2.9.1. The `metal`
extra is available when installing on another platform. The project pins
MuJoCo 3.10.0 and requires Python 3.12. Native Metal shader compilation
happens at runtime. Shader compilation may require Xcode and its separately
installed Metal Toolchain component; Command Line Tools alone may be
insufficient. The package does not build the MuJoCo core from source.

Check the environment without importing PyTorch:

```sh
mujoco-metal doctor
```

Run the optional native device smoke test with PyTorch's CPU fallback disabled:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 mujoco-metal doctor --gpu
```

Add `--json` to either command for machine-readable diagnostics. The GPU check
uses a tiny contact-free Euler model and reports only that narrow smoke test;
it does not qualify every model or MuJoCo feature.

## Development source

The **0.4.0** wheel contains the physics package and all shaders. Clone the
repository for the milestone demo scripts, XML models and GIFs, or to modify
the source. The following installs an editable checkout in an isolated environment:

```sh
git clone https://github.com/keeeeenw/mujoco-mac-metal.git
cd mujoco-mac-metal
git switch main
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './metal[metal,test]' pillow
export PYTORCH_ENABLE_MPS_FALLBACK=0
mujoco-metal doctor --gpu
```

For an existing checkout, fetch and update `main` before the
installation steps. Run demo commands from the repository root with this
environment active. `PYTHONPATH=metal` explicitly selects the checkout source.
Pillow is used only for recordings/plots; rendering still needs a working
MuJoCo OpenGL context. Headless physics checks do not need a viewer; offscreen
GIF recording does need OpenGL. For demos with a passive interactive viewer,
launch with `mjpython` rather than ordinary `python` on macOS.

The [gallery](examples/demo_gallery.md) links each runnable demo and its scope.
[Development coverage](DEVELOPMENT.md) is the current feature and qualification
record. Profiles reject unsupported combinations; installing this package does
not redirect arbitrary calls to `mujoco.mj_step` to Metal.

From the repository root, run CPU tests or opt into native GPU checks:

```sh
PYTHONPATH=metal python -m pytest -q metal/tests
MUJOCO_METAL_RUN_GPU=1 PYTHONPATH=metal python -m pytest -q metal/tests
```

Keep the GPU idle for qualification. The current source checkpoint passed 257
tests with GPU execution enabled. A separate environment without Torch passed
138 CPU tests, with 116 checks skipped. Optional Torch dependencies change
test collection, so the two totals are not directly additive.

## Local distribution build

From the repository root, build the current source version into a fresh
output directory and install that exact wheel in an isolated environment:

```sh
python -m pip install build
python -m build --outdir /tmp/mujoco-metal-dev-dist metal
python -m pip install --force-reinstall /tmp/mujoco-metal-dev-dist/mujoco_mac_metal-0.4.0-py3-none-any.whl
mujoco-metal doctor
PYTORCH_ENABLE_MPS_FALLBACK=0 mujoco-metal doctor --gpu
```

The filename follows `metal/pyproject.toml`; update it when changing versions.
The 0.4.0 wheel was installed in a clean Python 3.12 environment on the M1 Max
and passed the GPU doctor check plus all 257 package tests with native GPU
execution enabled and MPS fallback disabled. The installed module path was
checked to exclude the source checkout. All 39 packaged Python/shader files
matched the tested installation. The wheel includes all 15 shader resources. Building
or pushing source does not publish a release.

## PyPI release

Releases use PyPI Trusted Publishing, first configured for 0.3.0. The publisher uses
GitHub owner `keeeeenw`, repository `mujoco-mac-metal`, workflow
`publish-metal.yml`, and GitHub Actions environment `pypi`. For a new version, update the package and module versions, qualify the
built wheel on a supported Mac, then run the repository's **Publish Metal package to PyPI** workflow manually with
`workflow_dispatch`. It builds only the distribution under `metal/` and uses
GitHub Actions OIDC; no PyPI token is needed.
