# Install mujoco-mac-metal

The `mujoco-mac-metal` 0.3.0 release is not published on PyPI yet. Until it is
uploaded, install the current source from GitHub in a fresh Python 3.12
environment on an Apple Silicon Mac:

```sh
python3.12 -m venv .venv-metal
source .venv-metal/bin/activate
python -m pip install --upgrade pip
python -m pip install "mujoco-mac-metal[metal] @ git+https://github.com/keeeeenw/mujoco-mac-metal.git@main#subdirectory=metal"
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

## PyPI release

Version 0.3.0 is prepared as the `mujoco-mac-metal` distribution, but remains
unpublished until the owner uploads it. To publish, configure PyPI Trusted
Publishing for GitHub owner `keeeeenw`, repository `mujoco-mac-metal`, workflow
`publish-metal.yml`, and GitHub Actions environment `pypi`. Then run the
repository's **Publish Metal package to PyPI** workflow manually with
`workflow_dispatch`. It builds only the distribution under `metal/` and uses
GitHub Actions OIDC; no PyPI token is needed.

Maintainers can build and inspect the distributions locally from `metal/`:

```sh
python -m pip install build
python -m build
python -m pip install --force-reinstall dist/mujoco_mac_metal-0.3.0-py3-none-any.whl
mujoco-metal doctor
```
