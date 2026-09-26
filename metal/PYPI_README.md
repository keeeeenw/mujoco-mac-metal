# mujoco-mac-metal

An experimental generalized MuJoCo 3.10.0 stepping profile for Apple Silicon.
It provides contact-free native Metal stepping with applied forces, damping,
and scalar motors, exposed as the `mujoco_metal` Python module and
`mujoco-metal` command.

This is a narrow experimental profile. It does not implement contacts,
constraints, or the full MuJoCo feature set, and is not a replacement for
upstream MuJoCo. Visualization still uses upstream MuJoCo's OpenGL path.

Install on macOS arm64 with Python 3.12 using `pip install mujoco-mac-metal`.
The package selects PyTorch 2.9.1 by default on that platform. See the
[installation and doctor guide](https://github.com/keeeeenw/mujoco-mac-metal/blob/main/metal/INSTALL.md)
for qualification details and native GPU diagnostics. The
[project source](https://github.com/keeeeenw/mujoco-mac-metal/tree/main/metal)
contains examples and implementation details.
