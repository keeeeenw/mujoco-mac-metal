# Install the experimental MuJoCo Metal package

The package currently installs from the project repository. It is not yet a
published PyPI release. Use a fresh Python 3.12 environment on an Apple Silicon
Mac; the `metal` extra installs the pinned PyTorch dependency:

```sh
python3.12 -m venv .venv-metal
source .venv-metal/bin/activate
python -m pip install --upgrade pip
python -m pip install "mujoco-metal-experimental[metal] @ git+https://github.com/keeeeenw/mujoco-mac-metal.git@main#subdirectory=metal"
```

The first install downloads the project and its pinned dependencies. Native
Metal shader compilation happens at runtime. Some setups may also need Apple's
Xcode Command Line Tools and a current macOS Metal runtime. The package does
not build the MuJoCo core from source.

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

## PyPI status and release path

The current distribution name is `mujoco-metal-experimental`. The proposed
shorter PyPI name `mujoco-mac-metal` has not been published or reserved by this
project. A PyPI 404 only means no public project record was found at the time
checked; it does not establish whether a name is available. Until the project
owner confirms the name and publishes a release, use the repository install
command above.

The project already has a `pyproject.toml` entry point and includes Metal
shader files as package data, so a local wheel can be built and installed for
packaging checks. The examples currently live outside the installed Python
package. A release should first verify wheel and source distributions, shader
inclusion, the `mujoco-metal` command, and the documented demo workflow on a
clean supported Mac; run an early release through TestPyPI before a PyPI
release. The distribution name and public versioning policy remain project
decisions.

Maintainers can check the current wheel build locally from `metal/`:

```sh
python -m pip wheel --wheel-dir dist .
python -m pip install --force-reinstall dist/mujoco_metal_experimental-*.whl
mujoco-metal doctor
```

For a future automated release, configure PyPI Trusted Publishing for the
specific GitHub repository and release workflow, then use GitHub Actions OIDC
to publish. Do not put a PyPI API token in the repository or enable publishing
until an owner has confirmed the project name and release process. See the
[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [PyPI's Trusted Publisher setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
